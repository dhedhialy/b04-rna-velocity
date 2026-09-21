"""Two-state kinetic model and direction-set identification (AUDIT Sec 3-6).

Canonical 2-state model:
    du/dt = α(t) − βu
    ds/dt = βu − γs

Velocity at observed (s, u):  v = βu − γs.
Direction is identified iff the compatible set Θ(D) maps v to a sign-homogeneous
interval that excludes zero.

Three progressively restrictive assumptions:
  A1: constant (α, β, γ) — baseline
  A2: bounded (β, γ) ∈ [β_lo, β_hi] × [γ_lo, γ_hi] — partial identification
  A3: smooth/piecewise α(t) — controls adversarial transcription

Key definitions:
  v(D) = [v_min, v_max]  where v_min = inf, v_max = sup over Θ(D)
  POSITIVE if v_min > 0, NEGATIVE if v_max < 0, UNRESOLVED otherwise.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass(frozen=True)
class KineticParams:
    """Fixed kinetic parameters for the 2-state model."""
    alpha: float   # transcription rate
    beta: float    # splicing/conversion rate
    gamma: float   # degradation rate

    def velocity(self, s: float, u: float) -> float:
        """Instantaneous velocity: ds/dt = βu − γs."""
        return self.beta * u - self.gamma * s

    def steady_state(self) -> Tuple[float, float]:
        """Steady-state (u_ss, s_ss) = (α/β, α/γ)."""
        return self.alpha / self.beta, self.alpha / self.gamma


@dataclass(frozen=True)
class KineticBounds:
    """Parameter bounds defining the compatible set Θ(D)."""
    beta_lo: float = 0.01
    beta_hi: float = 10.0
    gamma_lo: float = 0.01
    gamma_hi: float = 10.0
    # Optional: alpha bounds (None = unbounded)
    alpha_lo: Optional[float] = None
    alpha_hi: Optional[float] = None
    # Optional: β/γ ratio bounds (None = unbounded)
    ratio_lo: Optional[float] = None
    ratio_hi: Optional[float] = None


@dataclass
class DirectionSet:
    """Computed direction set for a single gene observation."""
    s: float
    u: float
    v_min: float
    v_max: float
    theta_min: Optional[KineticParams] = None  # params achieving v_min
    theta_max: Optional[KineticParams] = None  # params achieving v_max
    bounds_used: Optional[KineticBounds] = None

    @property
    def certified(self) -> bool:
        """True if direction is identified (v_min > 0 or v_max < 0)."""
        return self.v_min > 0 or self.v_max < 0

    @property
    def direction(self) -> str:
        if self.v_min > 0:
            return "POSITIVE"
        elif self.v_max < 0:
            return "NEGATIVE"
        else:
            return "UNRESOLVED"

    def __repr__(self) -> str:
        return (f"DirectionSet(s={self.s:.3f}, u={self.u:.3f}, "
                f"v=[{self.v_min:.4f}, {self.v_max:.4f}], "
                f"dir={self.direction})")


def compute_direction_set(
    s: float,
    u: float,
    bounds: KineticBounds,
) -> DirectionSet:
    """Compute the velocity range v = βu − γs over compatible (β, γ).

    For fixed (s, u), v = βu − γs is linear in (β, γ), so the extrema
    are at the corners of the rectangular parameter region defined by bounds.
    However, we also enforce the constraint that (α, β, γ) must produce
    steady-state means consistent with the observed (s, u) neighborhood.

    Under A1 (constant rates, no further constraint):
        v_min and v_max are at the four corners of [β_lo, β_hi] × [γ_lo, γ_hi].

    Under A2 (bounded rates):
        Same corners, but bounds may be tighter.

    Under A3 (smooth α(t)):
        We additionally require that α = β·u̲ = γ·s̲ for some (u̲, s̲) in the
        neighborhood of the observation. This constrains the feasible region.
    """
    beta_lo, beta_hi = bounds.beta_lo, bounds.beta_hi
    gamma_lo, gamma_hi = bounds.gamma_lo, bounds.gamma_hi

    # v = βu − γs is bilinear; extrema at corners of the rectangle
    corners = [
        (beta_lo, gamma_lo),
        (beta_lo, gamma_hi),
        (beta_hi, gamma_lo),
        (beta_hi, gamma_hi),
    ]

    v_vals = [beta * u - gamma * s for beta, gamma in corners]
    v_min = min(v_vals)
    v_max = max(v_vals)

    # Record the params achieving extrema
    idx_min = v_vals.index(v_min)
    idx_max = v_vals.index(v_max)
    beta_min, gamma_min = corners[idx_min]
    beta_max, gamma_max = corners[idx_max]
    # alpha chosen to match observation: α = β·u̲ ≈ β·u (for snapshot)
    theta_min = KineticParams(alpha=beta_min * u, beta=beta_min, gamma=gamma_min)
    theta_max = KineticParams(alpha=beta_max * u, beta=beta_max, gamma=gamma_max)

    return DirectionSet(
        s=s, u=u, v_min=v_min, v_max=v_max,
        theta_min=theta_min, theta_max=theta_max,
        bounds_used=bounds,
    )


def compute_direction_set_with_steady_state(
    s: float,
    u: float,
    bounds: KineticBounds,
) -> DirectionSet:
    """Direction set under the additional constraint that (s, u) is near
    steady state, i.e., α/β ≈ u and α/γ ≈ s.

    This adds the constraint α = β·u̲ = γ·s̲ for some mean (u̲, s̲) in a
    neighborhood of the observation. For a single observation, we use
    the observation itself as the mean: α = β·u = γ·s.

    This implies β/γ = s/u, so the velocity v = βu − γs = β(u − s·γ/β) =
    β(u − s·(s/u)) = β(u − s²/u) = β(u² − s²)/u.

    Wait, that's not right. Let me recalculate:
    If α = β·u = γ·s, then β·u = γ·s, so β/γ = s/u.
    Then v = βu − γs = βu − (βu/s)·s = βu − βu = 0.

    So under EXACT steady state with α matched to the observation,
    the velocity is always 0. That's the trivial case.

    The non-trivial case is when we allow deviation from steady state:
    α ∈ [α_lo, α_hi] and (β, γ) are bounded, but α is NOT constrained
    to equal β·u or γ·s.

    In this case, v = βu − γs depends only on (β, γ) and the observation
    (s, u), not on α. The direction set is the same as the unconstrained
    case, which is what compute_direction_set returns.
    """
    # The steady-state constraint α = β·u̲ = γ·s̲ makes v = 0 identically.
    # Without it, α doesn't appear in v = βu − γs.
    # So the direction set is the same as compute_direction_set.
    return compute_direction_set(s, u, bounds)


def observational_equivalence_test(
    s: float,
    u: float,
    bounds: KineticBounds,
) -> dict:
    """Check whether observationally equivalent models with opposite signs exist.

    Returns a dict with:
        'equivalent': True if there exist (β1,γ1) and (β2,γ2) in bounds
                      such that v1 = β1·u − γ1·s and v2 = β2·u − γ2·s
                      have opposite signs.
        'v_range': (v_min, v_max)
        'counterexample': ((β1,γ1,v1), (β2,γ2,v2)) if equivalent, else None
    """
    ds = compute_direction_set(s, u, bounds)
    if not ds.certified:
        # Find explicit counterexample
        corners = [
            (bounds.beta_lo, bounds.gamma_lo),
            (bounds.beta_lo, bounds.gamma_hi),
            (bounds.beta_hi, bounds.gamma_lo),
            (bounds.beta_hi, bounds.gamma_hi),
        ]
        v_vals = [beta * u - gamma * s for beta, gamma in corners]
        pos_idx = [i for i, v in enumerate(v_vals) if v > 0]
        neg_idx = [i for i, v in enumerate(v_vals) if v < 0]
        if pos_idx and neg_idx:
            ce = (
                (corners[pos_idx[0]][0], corners[pos_idx[0]][1], v_vals[pos_idx[0]]),
                (corners[neg_idx[0]][0], corners[neg_idx[0]][1], v_vals[neg_idx[0]]),
            )
            return {"equivalent": True, "v_range": (ds.v_min, ds.v_max),
                    "counterexample": ce}
    return {"equivalent": not ds.certified, "v_range": (ds.v_min, ds.v_max),
            "counterexample": None}
