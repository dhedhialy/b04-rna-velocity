"""Validation framework: type-I error, coverage, power (AUDIT Sec 12).

Validates the direction certificate against:
1. Synthetic ground truth (where direction is known)
2. Held-out labeling times (where prediction is testable)
3. Negative controls (where direction should be UNRESOLVED)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .kinetic import KineticParams, KineticBounds
from .identifiability import identify_direction, IdentificationResult
from .simulation import (
    generate_adversarial_pair, simulate_counts,
    simulate_labeling, labeling_residual,
)


@dataclass
class ValidationMetrics:
    """Aggregate validation metrics."""
    n_genes: int = 0
    n_certified: int = 0
    n_unresolved: int = 0

    # Type-I error: certified directions that are wrong
    n_false_positive: int = 0  # certified POSITIVE but true is NEGATIVE
    n_false_negative: int = 0  # certified NEGATIVE but true is POSITIVE
    type_i_error: float = 0.0

    # Coverage: fraction of true directions within [v_min, v_max]
    n_covered: int = 0
    coverage: float = 0.0

    # Power: fraction of truly non-zero directions that are certified
    n_truly_nonzero: int = 0
    n_correctly_certified: int = 0
    power: float = 0.0

    # Abstention: fraction of truly zero directions that are UNRESOLVED
    n_truly_zero: int = 0
    n_correctly_abstained: int = 0
    abstention_rate: float = 0.0


def synthetic_population(
    bounds: KineticBounds,
    n_genes: int = 200,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Build a designed synthetic population with unambiguous ground truth.

    Regime split (roughly thirds):
      * INDUCED:   u/s ratio 2-4x above γ_hi/β_lo  -> true_v > 0 for every
                   (β, γ) in bounds -> certificate MUST certify POSITIVE.
      * REPRESSED: u/s ratio 2-4x below γ_lo/β_hi  -> true_v < 0 for every
                   (β, γ) in bounds -> certificate MUST certify NEGATIVE.
      * AMBIENT:   u/s ratio inside [γ_lo/β_hi, γ_hi/β_lo] -> compatible
                   models have mixed signs -> certificate MUST abstain.

    Each row carries its own ground truth (true_v, regime) plus the
    certificate verdict, so validators and certificate issuance read the
    SAME genes (single source of truth -- no stale random demo clouds).
    """
    rng = np.random.default_rng(seed)
    r_lo = bounds.gamma_lo / bounds.beta_hi
    r_hi = bounds.gamma_hi / bounds.beta_lo
    n_ind = n_genes // 3
    n_rep = n_genes // 3
    n_amb = n_genes - n_ind - n_rep
    ratio_ind = r_hi * np.exp(rng.uniform(np.log(2.0), np.log(4.0), n_ind))
    ratio_rep = r_lo / np.exp(rng.uniform(np.log(2.0), np.log(4.0), n_rep))
    ratio_amb = r_lo + (r_hi - r_lo) * rng.uniform(0.1, 0.9, n_amb)
    ratios = np.concatenate([ratio_ind, ratio_rep, ratio_amb])
    regime = (["induced"] * n_ind + ["repressed"] * n_rep
              + ["ambient"] * n_amb)
    rng_obs = np.random.default_rng(seed + 1_000_000)  # independent, but pinned

    rows = []
    for i in range(len(regime)):
        beta = float(rng_obs.uniform(bounds.beta_lo, bounds.beta_hi))
        gamma = float(rng_obs.uniform(bounds.gamma_lo, bounds.gamma_hi))
        s_obs = float(rng_obs.uniform(5.0, 50.0))
        u_obs = max(0.1, s_obs * ratios[i])
        true_v = beta * u_obs - gamma * s_obs
        id_result = identify_direction(s_obs, u_obs, bounds)
        rows.append({
            "gene_id": f"G{i:04d}", "regime": regime[i], "s": s_obs, "u": u_obs,
            "true_v": true_v, "beta": beta, "gamma": gamma,
            "direction": id_result.direction, "certified": id_result.certified,
            "v_min": id_result.v_range[0], "v_max": id_result.v_range[1],
            "width": id_result.v_range[1] - id_result.v_range[0],
        })
    return rows


def validate_on_synthetic(
    bounds: KineticBounds,
    n_genes: int = 200,
    seed: int = 0,
) -> ValidationMetrics:
    """Validate direction certificates on a designed synthetic population.

    The population is DESIGNED to make the certificate's invariants testable
    (not a random near-steady-state cloud, where everything is silently
    UNRESOLVED and -- as the bug we fixed showed -- the metrics read 0.0).
    See `synthetic_population` for the regime design.

    The meaningful invariants checked:

      * Coverage      == 1.0   (the [v_min, v_max] envelope over a rectangle
                                always contains the true velocity; a linear
                                consequence of v = βu − γs).
      * Type-I error  == 0.0   (a certified sign can never contradict the
                                truth: certification requires ALL compatible
                                models -- including the true one -- to agree).
      * Recall (IND/REP) == 1.0 and Abstention (AMBIENT) == 1.0.
    """
    metrics = ValidationMetrics()

    for row in synthetic_population(bounds, n_genes, seed):
        true_v = row["true_v"]
        id_cert = row["certified"]
        direction = row["direction"]
        regime = row["regime"]

        metrics.n_genes += 1
        if id_cert:
            metrics.n_certified += 1
        else:
            metrics.n_unresolved += 1

        if regime in ("induced", "repressed"):
            assert (regime == "induced") == (true_v > 0), "regime design broken"
            metrics.n_truly_nonzero += 1
            correct = (regime == "induced" and direction == "POSITIVE") or \
                      (regime == "repressed" and direction == "NEGATIVE")
            if correct:
                metrics.n_correctly_certified += 1
            if direction == "NEGATIVE" and regime == "induced":
                metrics.n_false_negative += 1
            if direction == "POSITIVE" and regime == "repressed":
                metrics.n_false_positive += 1
        else:
            metrics.n_truly_zero += 1
            if direction == "UNRESOLVED":
                metrics.n_correctly_abstained += 1

        # Coverage MUST hold by the linear-envelope identity (true params in-bounds)
        if row["v_min"] <= true_v <= row["v_max"]:
            metrics.n_covered += 1

    if metrics.n_genes > 0:
        metrics.coverage = metrics.n_covered / metrics.n_genes
    if metrics.n_truly_nonzero > 0:
        metrics.power = metrics.n_correctly_certified / metrics.n_truly_nonzero
    if metrics.n_truly_zero > 0:
        metrics.abstention_rate = metrics.n_correctly_abstained / metrics.n_truly_zero
    total_errors = metrics.n_false_positive + metrics.n_false_negative
    if metrics.n_certified > 0:
        metrics.type_i_error = total_errors / metrics.n_certified

    return metrics


def validate_on_counterexamples(
    bounds: KineticBounds,
    n_pairs: int = 50,
    seed: int = 0,
) -> Dict[str, float]:
    """Validate that adversarial pairs are correctly identified as UNRESOLVED.

    For each observationally equivalent pair (same (s, u), models with
    v_A > 0 and v_B < 0), the certificate is REQUIRED to abstain: if the
    compatible set contains opposite signs, the direction cannot be
    identified. Uses the counterexample's own (s, u) -- NOT a fixed probe
    point -- so the check is non-circular.
    """
    from .identifiability import generate_counterexample

    n_abstained = 0
    n_total = 0
    rng = np.random.default_rng(seed)

    for i in range(n_pairs):
        s = float(rng.uniform(5.0, 50.0))
        u = float(rng.uniform(5.0, 50.0))
        ce = generate_counterexample(s=s, u=u, bounds=bounds)
        if ce is None:
            continue  # (s, u) falls outside the unresolved region; not a counterexample
        n_total += 1
        id_result = identify_direction(s, u, bounds)
        # The counterexample PROVES opposite-sign models fit this snapshot,
        # so the certificate must abstain.
        if id_result.direction == "UNRESOLVED":
            n_abstained += 1

    return {
        "n_pairs": n_total,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n_total if n_total > 0 else 0.0,
    }


def validate_labeling_power(
    bounds: KineticBounds,
    n_genes: int = 100,
    t_label: float = 0.5,
    seed: int = 0,
) -> Dict[str, float]:
    """Validate that labeling tests have power against adversarial models.

    Generates adversarial pairs where snapshot-compatible models have
    opposite directions, then checks if labeling correctly separates them.
    """
    rng = np.random.default_rng(seed)
    n_separated = 0
    n_total = 0

    for i in range(n_genes):
        # Random kinetic parameters for model A
        beta_a = rng.uniform(bounds.beta_lo, bounds.beta_hi)
        gamma_a = rng.uniform(bounds.gamma_lo, bounds.gamma_hi)

        # Model B: opposite direction (swap β/γ ratio)
        beta_b = gamma_a * rng.uniform(0.8, 1.2)
        gamma_b = beta_a * rng.uniform(0.8, 1.2)

        # Observation near steady state of model A
        u_ss = rng.uniform(5.0, 50.0)
        s_ss = u_ss * (beta_a / gamma_a)
        s_obs = max(0.1, s_ss * (1 + rng.normal(0, 0.1)))
        u_obs = max(0.1, u_ss * (1 + rng.normal(0, 0.1)))

        # Labeling residuals for both models
        params_a = KineticParams(alpha=beta_a * u_obs, beta=beta_a, gamma=gamma_a)
        params_b = KineticParams(alpha=beta_b * u_obs, beta=beta_b, gamma=gamma_b)

        # Simulate labeling (ground truth from model A)
        labeled = simulate_labeling(params_a, t_label, n_cells=100, seed=seed + i)
        labeled_s = float(np.mean(labeled[:, 0]))
        labeled_u = float(np.mean(labeled[:, 1]))

        r_a = labeling_residual(params_a, t_label, labeled_s, labeled_u)
        r_b = labeling_residual(params_b, t_label, labeled_s, labeled_u)

        n_total += 1
        if abs(r_a - r_b) > 0.1:  # labeling separates the models
            n_separated += 1

    return {
        "n_genes": n_total,
        "n_separated": n_separated,
        "power": n_separated / n_total if n_total > 0 else 0.0,
    }
