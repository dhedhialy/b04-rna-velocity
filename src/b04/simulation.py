"""Synthetic data generation for counterexample and power analysis (AUDIT Sec 10).

Generates paired kinetic models (θ_A, θ_B) that are observationally equivalent
under snapshot observation but produce opposite velocity signs, then adds
metabolic-labeling constraints to show when the two models separate.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .kinetic import KineticParams, KineticBounds


@dataclass
class AdversarialPair:
    """A pair of observationally equivalent models with opposite velocity signs."""
    theta_a: KineticParams
    theta_b: KineticParams
    s: float         # observed spliced count
    u: float         # observed unspliced count
    v_a: float       # velocity under model A
    v_b: float       # velocity under model B
    label_a: str = "inducing"   # biological label for model A
    label_b: str = "repressing" # biological label for model B


def generate_adversarial_pair(
    s_range: Tuple[float, float] = (1.0, 100.0),
    u_range: Tuple[float, float] = (1.0, 100.0),
    rate_range: Tuple[float, float] = (0.1, 5.0),
    seed: Optional[int] = None,
    n_attempts: int = 1000,
) -> Optional[AdversarialPair]:
    """Generate a single adversarial pair.

    Strategy: pick (s, u) in the unresolved region, then find two (β, γ)
    pairs at opposite corners of the parameter rectangle that give
    opposite velocity signs.

    The velocity is v = βu − γs. For a given (s, u):
    - Model A: (β_hi, γ_lo) → v_A = β_hi·u − γ_lo·s > 0 (high β, low γ)
    - Model B: (β_lo, γ_hi) → v_B = β_lo·u − γ_hi·s < 0 (low β, high γ)

    Both models are observationally equivalent in the sense that they
    produce counts in the same range (same Poisson means), but the
    velocity signs differ.
    """
    rng = np.random.default_rng(seed)

    for _ in range(n_attempts):
        s = rng.uniform(*s_range)
        u = rng.uniform(*u_range)

        # Pick rate bounds
        beta_lo = rng.uniform(0.05, 0.5)
        beta_hi = rng.uniform(2.0, 10.0)
        gamma_lo = rng.uniform(0.05, 0.5)
        gamma_hi = rng.uniform(2.0, 10.0)

        # Check unresolved condition: γ_lo/β_hi ≤ u/s ≤ γ_hi/β_lo
        r_lo = gamma_lo / beta_hi
        r_hi = gamma_hi / beta_lo
        ratio = u / s

        if r_lo <= ratio <= r_hi:
            # Model A: high β, low γ → positive velocity
            v_a = beta_hi * u - gamma_lo * s
            # Model B: low β, high γ → negative velocity
            v_b = beta_lo * u - gamma_hi * s

            if v_a > 0 and v_b < 0:
                theta_a = KineticParams(alpha=beta_hi * u, beta=beta_hi, gamma=gamma_lo)
                theta_b = KineticParams(alpha=beta_lo * u, beta=beta_lo, gamma=gamma_hi)
                return AdversarialPair(
                    theta_a=theta_a, theta_b=theta_b,
                    s=s, u=u, v_a=v_a, v_b=v_b,
                )
    return None


def generate_counterexample_family(
    n_pairs: int = 50,
    s_range: Tuple[float, float] = (1.0, 100.0),
    u_range: Tuple[float, float] = (1.0, 100.0),
    rate_range: Tuple[float, float] = (0.1, 5.0),
    seed: int = 0,
) -> List[AdversarialPair]:
    """Generate a family of adversarial pairs for visualization."""
    pairs = []
    for i in range(n_pairs):
        p = generate_adversarial_pair(s_range, u_range, rate_range, seed=seed + i)
        if p is not None:
            pairs.append(p)
    return pairs


def simulate_counts(
    params: KineticParams,
    n_cells: int = 500,
    n_time: int = 1,
    dt: float = 0.1,
    n_genes: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate spliced/unspliced counts from the 2-state model.

    Uses Euler-Maruyama integration of the ODE system:
        du/dt = α − βu
        ds/dt = βu − γs

    Then adds Poisson noise to generate count observations.

    Returns: (n_cells, n_genes, 2) array of [s, u] counts.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(n_time / dt)

    # Initialize from steady state
    u_ss = params.alpha / params.beta
    s_ss = params.alpha / params.gamma

    # Simulate n_cells independent trajectories
    u = np.full(n_cells, u_ss) + rng.normal(0, 0.1 * u_ss, n_cells)
    s = np.full(n_cells, s_ss) + rng.normal(0, 0.1 * s_ss, n_cells)
    u = np.maximum(u, 0.01)
    s = np.maximum(s, 0.01)

    # Integrate
    for _ in range(n_steps):
        du = params.alpha - params.beta * u
        ds = params.beta * u - params.gamma * s
        u = u + du * dt
        s = s + ds * dt
        u = np.maximum(u, 0.0)
        s = np.maximum(s, 0.0)

    # Poisson noise (discretize to counts)
    u_counts = rng.poisson(np.maximum(u, 0.1))
    s_counts = rng.poisson(np.maximum(s, 0.1))

    return np.stack([s_counts, u_counts], axis=-1)


def simulate_labeling(
    params: KineticParams,
    t_label: float,
    n_cells: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate metabolic labeling at time t_label.

    The labeled fraction of spliced RNA at time t_label is approximately:
        F_labeled ≈ 1 − exp(−γ · t_label)    (for spliced)
        F_labeled_u ≈ 1 − exp(−β · t_label)  (for unspliced)

    Returns: (n_cells, 2) array of [labeled_s, labeled_u] fractions.
    """
    rng = np.random.default_rng(seed)

    # Labeled fractions (deterministic under the model)
    f_s = 1.0 - np.exp(-params.gamma * t_label)
    f_u = 1.0 - np.exp(-params.beta * t_label)

    # Add noise (biological variability + measurement)
    labeled_s = rng.beta(f_s * 10, (1 - f_s) * 10, n_cells)
    labeled_u = rng.beta(f_u * 10, (1 - f_u) * 10, n_cells)

    return np.stack([labeled_s, labeled_u], axis=-1)


def labeling_residual(
    params: KineticParams,
    t_label: float,
    observed_labeled_s: float,
    observed_labeled_u: float,
) -> float:
    """Compute the labeling residual R_g = observed − predicted.

    Under the kinetic model, the expected labeled fractions are:
        F_s = 1 − exp(−γ · t_label)
        F_u = 1 − exp(−β · t_label)

    The residual measures how well the snapshot-compatible model
    predicts the labeling observation.
    """
    predicted_s = 1.0 - np.exp(-params.gamma * t_label)
    predicted_u = 1.0 - np.exp(-params.beta * t_label)
    return np.sqrt((observed_labeled_s - predicted_s)**2 +
                   (observed_labeled_u - predicted_u)**2)
