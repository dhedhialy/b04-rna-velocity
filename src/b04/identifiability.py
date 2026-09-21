"""Identification theorems and counterexample generation (AUDIT Sec 5-6, 10).

Theorem 1 (Sufficient condition for direction identification).
    Let (s, u) be observed steady-state counts for a gene. Let
    Θ(D) = {(β, γ) : β ∈ [β_lo, β_hi], γ ∈ [γ_lo, γ_hi]} be the
    compatible parameter set. Then:

    (a) If u/s > γ_hi/β_lo, then v = βu − γs > 0 for all (β, γ) ∈ Θ(D),
        so the direction is POSITIVE (the gene is being induced).

    (b) If u/s < γ_lo/β_hi, then v = βu − γs < 0 for all (β, γ) ∈ Θ(D),
        so the direction is NEGATIVE (the gene is being repressed).

    (c) If γ_lo/β_hi ≤ u/s ≤ γ_hi/β_lo, the direction is UNRESOLVED:
        there exist compatible parameters giving both signs.

    The width of the unresolved region is:
        W = γ_hi/β_lo − γ_lo/β_hi

    so identification improves (W shrinks) as rate bounds tighten.

Theorem 2 (Necessary condition for identification).
    The direction is identified for ALL observations only if the parameter
    region Θ(D) is a singleton (β, γ known exactly). In particular:
    - Bounded but non-degenerate Θ(D) always admits UNRESOLVED observations.
    - The fraction of unresolved observations is at least
      W / (W + max(u/s)) in a population of genes.

Theorem 3 (Metabolic-labeling refinement).
    Let t_label be a metabolic-labeling time. If the labeling data constrain
    the kinetic rates such that the effective Θ(D, t_label) has
    W' < W, then labeling provably shrinks the unresolved set. If labeling
    reduces W' to 0 for a gene, that gene's direction is identified.

Counterexample construction:
    Given any observation (s, u) with γ_lo/β_hi ≤ u/s ≤ γ_hi/β_lo,
    the four corners of the parameter rectangle give explicit
    observationally equivalent models with opposite velocity signs.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .kinetic import KineticBounds, DirectionSet, compute_direction_set


@dataclass(frozen=True)
class IdentificationResult:
    """Result of applying Theorem 1 to a single gene observation."""
    s: float
    u: float
    ratio_su: float          # u/s (the observable ratio)
    ratio_bounds: Tuple[float, float]  # (γ_lo/β_hi, γ_hi/β_lo)
    certified: bool
    direction: str           # POSITIVE, NEGATIVE, UNRESOLVED
    v_range: Tuple[float, float]
    width_unresolved: float  # W = γ_hi/β_lo − γ_lo/β_hi


def identify_direction(
    s: float,
    u: float,
    bounds: KineticBounds,
) -> IdentificationResult:
    """Apply Theorem 1 to determine if direction is identified."""
    if s <= 0 or u <= 0:
        # Edge case: non-positive counts. Velocity sign depends on which is zero.
        if u <= 0 and s > 0:
            return IdentificationResult(
                s=s, u=u, ratio_su=0.0,
                ratio_bounds=(bounds.gamma_lo / bounds.beta_hi,
                              bounds.gamma_hi / bounds.beta_lo),
                certified=True, direction="NEGATIVE",
                v_range=(-np.inf, -bounds.gamma_lo * s),
                width_unresolved=bounds.gamma_hi / bounds.beta_lo - bounds.gamma_lo / bounds.beta_hi,
            )
        elif s <= 0 and u > 0:
            return IdentificationResult(
                s=s, u=u, ratio_su=np.inf,
                ratio_bounds=(bounds.gamma_lo / bounds.beta_hi,
                              bounds.gamma_hi / bounds.beta_lo),
                certified=True, direction="POSITIVE",
                v_range=(bounds.beta_lo * u, np.inf),
                width_unresolved=bounds.gamma_hi / bounds.beta_lo - bounds.gamma_lo / bounds.beta_hi,
            )
        else:
            return IdentificationResult(
                s=s, u=u, ratio_su=float('nan'),
                ratio_bounds=(bounds.gamma_lo / bounds.beta_hi,
                              bounds.gamma_hi / bounds.beta_lo),
                certified=False, direction="UNRESOLVED",
                v_range=(0.0, 0.0),
                width_unresolved=bounds.gamma_hi / bounds.beta_lo - bounds.gamma_lo / bounds.beta_hi,
            )

    ratio_su = u / s
    r_lo = bounds.gamma_lo / bounds.beta_hi  # lower bound on γ/β
    r_hi = bounds.gamma_hi / bounds.beta_lo  # upper bound on γ/β
    width = r_hi - r_lo

    ds = compute_direction_set(s, u, bounds)

    if ratio_su > r_hi:
        direction = "POSITIVE"
        certified = True
    elif ratio_su < r_lo:
        direction = "NEGATIVE"
        certified = True
    else:
        direction = "UNRESOLVED"
        certified = False

    return IdentificationResult(
        s=s, u=u, ratio_su=ratio_su,
        ratio_bounds=(r_lo, r_hi),
        certified=certified, direction=direction,
        v_range=(ds.v_min, ds.v_max),
        width_unresolved=width,
    )


def generate_counterexample(
    s: float,
    u: float,
    bounds: KineticBounds,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Generate an explicit observational-equivalence counterexample.

    If γ_lo/β_hi ≤ u/s ≤ γ_hi/β_lo (UNRESOLVED region), returns two
    parameter vectors (β, γ, v) with opposite velocity signs.
    """
    r_lo = bounds.gamma_lo / bounds.beta_hi
    r_hi = bounds.gamma_hi / bounds.beta_lo

    if r_lo <= u / s <= r_hi:
        # Two corners with opposite signs
        # Corner 1: (β_hi, γ_lo) → v = β_hi·u − γ_lo·s > 0 (high β, low γ)
        # Corner 2: (β_lo, γ_hi) → v = β_lo·u − γ_hi·s < 0 (low β, high γ)
        v1 = bounds.beta_hi * u - bounds.gamma_lo * s
        v2 = bounds.beta_lo * u - bounds.gamma_hi * s
        if v1 > 0 and v2 < 0:
            return (
                (bounds.beta_hi, bounds.gamma_lo, v1),
                (bounds.beta_hi, bounds.gamma_hi, v2),
            )
        # Fallback: try all four corners
        corners = [
            (bounds.beta_lo, bounds.gamma_lo),
            (bounds.beta_lo, bounds.gamma_hi),
            (bounds.beta_hi, bounds.gamma_lo),
            (bounds.beta_hi, bounds.gamma_hi),
        ]
        v_vals = [beta * u - gamma * s for beta, gamma in corners]
        pos = [(beta, gamma, v) for (beta, gamma), v in zip(corners, v_vals) if v > 0]
        neg = [(beta, gamma, v) for (beta, gamma), v in zip(corners, v_vals) if v < 0]
        if pos and neg:
            return (pos[0], neg[0])
    return None


def batch_identify(
    observations: np.ndarray,
    bounds: KineticBounds,
) -> List[IdentificationResult]:
    """Apply Theorem 1 to a batch of (s, u) observations.

    Args:
        observations: Nx2 array of (s, u) values.
        bounds: kinetic parameter bounds.

    Returns:
        List of IdentificationResult, one per observation.
    """
    results = []
    for s, u in observations:
        results.append(identify_direction(float(s), float(u), bounds))
    return results


def unresolved_fraction(
    observations: np.ndarray,
    bounds: KineticBounds,
) -> dict:
    """Compute fraction of unresolved observations in a population.

    Returns dict with counts and fractions, plus the theoretical lower
    bound from Theorem 2.
    """
    results = batch_identify(observations, bounds)
    n = len(results)
    n_pos = sum(1 for r in results if r.direction == "POSITIVE")
    n_neg = sum(1 for r in results if r.direction == "NEGATIVE")
    n_unr = sum(1 for r in results if r.direction == "UNRESOLVED")

    # Theorem 2 lower bound on unresolved fraction
    r_lo = bounds.gamma_lo / bounds.beta_hi
    r_hi = bounds.gamma_hi / bounds.beta_lo
    width = r_hi - r_lo
    # Empirical ratio distribution
    ratios = np.array([r.ratio_su for r in results if np.isfinite(r.ratio_su)])
    if len(ratios) > 0:
        max_ratio = np.percentile(ratios, 95)  # 95th percentile as proxy
        theoretical_min_unresolved = width / (width + max_ratio)
    else:
        theoretical_min_unresolved = 0.0

    return {
        "n_total": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_unresolved": n_unr,
        "fraction_unresolved": n_unr / n if n > 0 else 0.0,
        "fraction_positive": n_pos / n if n > 0 else 0.0,
        "fraction_negative": n_neg / n if n > 0 else 0.0,
        "width_unresolved_region": width,
        "theoretical_min_unresolved": theoretical_min_unresolved,
    }
