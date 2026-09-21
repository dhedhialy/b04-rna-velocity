"""Metabolic-labeling falsification test (AUDIT Sec 9).

The key test is not "does labeling correlate with velocity?" but:
Does the certificate correctly abstain or retain direction when an
independent labeling observation makes the kinetic assumptions testable?

A successful test should have calibrated type-I error and useful power
against adversarial kinetic alternatives that match snapshot behavior
but diverge at held-out labeling times.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .kinetic import KineticParams, KineticBounds
from .simulation import labeling_residual


@dataclass
class LabelingTestResult:
    """Result of a metabolic-labeling falsification test."""
    gene_id: str
    n_cells: int
    t_label: float

    # Snapshot-compatible models
    n_compatible_models: int
    n_models_passing_labeling: int
    fraction_passing: float

    # Residual statistics
    residual_mean: float
    residual_std: float
    residual_min: float
    residual_max: float

    # Calibration (null distribution)
    null_residual_mean: Optional[float] = None
    null_residual_std: Optional[float] = None
    p_value: Optional[float] = None

    # Decision
    test_passed: Optional[bool] = None
    direction_validated: Optional[bool] = None


def labeling_falsification_test(
    gene_id: str,
    s_obs: float,
    u_obs: float,
    n_cells: int,
    t_label: float,
    bounds: KineticBounds,
    observed_labeled_s: float,
    observed_labeled_u: float,
    n_models: int = 100,
    n_null: int = 1000,
    seed: int = 0,
) -> LabelingTestResult:
    """Run the metabolic-labeling falsification test.

    1. Sample compatible models from Θ(D).
    2. Compute labeling residual for each model.
    3. Compare to null distribution (residual under model misspecification).
    4. If most compatible models fail the labeling test, the direction
       certificate is falsified.

    Args:
        gene_id: gene identifier.
        s_obs, u_obs: observed steady-state counts.
        n_cells: number of cells in the observation.
        t_label: metabolic-labeling time.
        bounds: kinetic parameter bounds.
        observed_labeled_s, observed_labeled_u: observed labeled fractions.
        n_models: number of compatible models to sample.
        n_null: number of null draws for calibration.
        seed: random seed.
    """
    rng = np.random.default_rng(seed)

    # Sample compatible models from the parameter rectangle
    betas = rng.uniform(bounds.beta_lo, bounds.beta_hi, n_models)
    gammas = rng.uniform(bounds.gamma_lo, bounds.gamma_hi, n_models)

    # Compute labeling residual for each model
    residuals = []
    for beta, gamma in zip(betas, gammas):
        params = KineticParams(alpha=beta * u_obs, beta=beta, gamma=gamma)
        r = labeling_residual(params, t_label, observed_labeled_s, observed_labeled_u)
        residuals.append(r)
    residuals = np.array(residuals)

    # Null distribution: residuals under random parameter draws
    # (representing model misspecification)
    null_betas = rng.uniform(bounds.beta_lo, bounds.beta_hi, n_null)
    null_gammas = rng.uniform(bounds.gamma_lo, bounds.gamma_hi, n_null)
    null_residuals = []
    for beta, gamma in zip(null_betas, null_gammas):
        params = KineticParams(alpha=beta * s_obs, beta=beta, gamma=gamma)
        # Use a "wrong" observation (shuffled) as null
        r = labeling_residual(params, t_label,
                              rng.uniform(0, 1), rng.uniform(0, 1))
        null_residuals.append(r)
    null_residuals = np.array(null_residuals)

    # P-value: fraction of null residuals smaller than observed mean
    p_value = float(np.mean(null_residuals <= residuals.mean()))

    # Decision: test passes if most compatible models have small residuals
    threshold = np.percentile(null_residuals, 95)
    n_passing = int(np.sum(residuals < threshold))
    fraction_passing = n_passing / n_models

    test_passed = fraction_passing > 0.5

    return LabelingTestResult(
        gene_id=gene_id,
        n_cells=n_cells,
        t_label=t_label,
        n_compatible_models=n_models,
        n_models_passing_labeling=n_passing,
        fraction_passing=fraction_passing,
        residual_mean=float(residuals.mean()),
        residual_std=float(residuals.std()),
        residual_min=float(residuals.min()),
        residual_max=float(residuals.max()),
        null_residual_mean=float(null_residuals.mean()),
        null_residual_std=float(null_residuals.std()),
        p_value=p_value,
        test_passed=test_passed,
        direction_validated=test_passed,
    )


def batch_labeling_test(
    genes: List[dict],
    t_label: float,
    bounds: KineticBounds,
    n_models: int = 100,
    n_null: int = 1000,
    seed: int = 0,
) -> List[LabelingTestResult]:
    """Run labeling test on a batch of genes.

    Each gene dict should have:
        gene_id, s_obs, u_obs, n_cells, labeled_s, labeled_u
    """
    results = []
    for i, gene in enumerate(genes):
        result = labeling_falsification_test(
            gene_id=gene["gene_id"],
            s_obs=gene["s_obs"],
            u_obs=gene["u_obs"],
            n_cells=gene["n_cells"],
            t_label=t_label,
            bounds=bounds,
            observed_labeled_s=gene["labeled_s"],
            observed_labeled_u=gene["labeled_u"],
            n_models=n_models,
            n_null=n_null,
            seed=seed + i,
        )
        results.append(result)
    return results
