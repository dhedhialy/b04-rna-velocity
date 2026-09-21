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


def validate_on_synthetic(
    bounds: KineticBounds,
    n_genes: int = 200,
    seed: int = 0,
) -> ValidationMetrics:
    """Validate direction certificates on synthetic data with known ground truth.

    Generates genes with known velocity directions (from the kinetic model),
    then checks whether the certificate correctly identifies or abstains.
    """
    rng = np.random.default_rng(seed)
    metrics = ValidationMetrics()

    for i in range(n_genes):
        # Random kinetic parameters
        beta = rng.uniform(bounds.beta_lo, bounds.beta_hi)
        gamma = rng.uniform(bounds.gamma_lo, bounds.gamma_hi)

        # Random observation near steady state
        u_ss = rng.uniform(1.0, 50.0)
        s_ss = u_ss * (beta / gamma)  # steady state: s = u * β/γ

        # Add noise
        s_obs = max(0.1, s_ss + rng.normal(0, 0.5))
        u_obs = max(0.1, u_ss + rng.normal(0, 0.5))

        # True velocity
        true_v = beta * u_ss - gamma * s_ss  # = 0 at exact steady state
        # Off-steady-state: perturb
        s_obs = max(0.1, s_ss * (1 + rng.normal(0, 0.3)))
        u_obs = max(0.1, u_ss * (1 + rng.normal(0, 0.3)))
        true_v = beta * u_obs - gamma * s_obs

        # Apply certificate
        id_result = identify_direction(s_obs, u_obs, bounds)

        # Count
        metrics.n_genes += 1
        if id_result.certified:
            metrics.n_certified += 1
        else:
            metrics.n_unresolved += 1

        # Check correctness
        if true_v > 0:
            metrics.n_truly_nonzero += 1
            if id_result.direction == "POSITIVE":
                metrics.n_correctly_certified += 1
            elif id_result.direction == "NEGATIVE":
                metrics.n_false_negative += 1
        elif true_v < 0:
            metrics.n_truly_nonzero += 1
            if id_result.direction == "NEGATIVE":
                metrics.n_correctly_certified += 1
            elif id_result.direction == "POSITIVE":
                metrics.n_false_positive += 1
        else:
            metrics.n_truly_zero += 1
            if id_result.direction == "UNRESOLVED":
                metrics.n_correctly_abstained += 1

            # Coverage
            if id_result.v_range[0] <= true_v <= id_result.v_range[1]:
                metrics.n_covered += 1

    # Compute rates
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

    The direction certificate should ABSTAIN for observationally equivalent
    models with opposite velocity signs.
    """
    from .identifiability import generate_counterexample

    n_abstained = 0
    n_total = 0

    for i in range(n_pairs):
        ce = generate_counterexample(
            s=25.0, u=25.0,
            bounds=bounds,
        )
        if ce is not None:
            n_total += 1
            # The observation (s, u) from the counterexample
            # should be UNRESOLVED
            s = 25.0  # midpoint of range
            u = 25.0
            id_result = identify_direction(s, u, bounds)
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
