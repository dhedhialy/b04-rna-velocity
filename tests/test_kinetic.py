"""Tests for the kinetic model and identification framework."""

import numpy as np
import pytest

from b04.kinetic import KineticParams, KineticBounds, DirectionSet, compute_direction_set
from b04.identifiability import (
    identify_direction, generate_counterexample,
    batch_identify, unresolved_fraction,
)
from b04.simulation import (
    generate_adversarial_pair, generate_counterexample_family,
    simulate_counts, simulate_labeling, labeling_residual,
)
from b04.certificate import build_certificate, Certificate
from b04.labeling import labeling_falsification_test
from b04.validation import (
    validate_on_synthetic, validate_on_counterexamples,
    validate_labeling_power,
)


class TestKineticModel:
    def test_velocity_formula(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.3)
        assert params.velocity(s=2.0, u=3.0) == pytest.approx(0.5 * 3.0 - 0.3 * 2.0)

    def test_steady_state(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.2)
        u_ss, s_ss = params.steady_state()
        assert u_ss == pytest.approx(2.0)
        assert s_ss == pytest.approx(5.0)

    def test_velocity_at_steady_state(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.2)
        u_ss, s_ss = params.steady_state()
        assert params.velocity(s_ss, u_ss) == pytest.approx(0.0)


class TestDirectionSet:
    def test_positive_direction(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=1.0)
        # u/s = 100/1 = 100, γ_hi/β_lo = 1.0/0.1 = 10 → u/s > r_hi → POSITIVE
        ds = compute_direction_set(s=1.0, u=100.0, bounds=bounds)
        assert ds.direction == "POSITIVE"
        assert ds.certified

    def test_negative_direction(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=1.0, gamma_hi=10.0)
        # u/s = 1/100 = 0.01, γ_lo/β_hi = 1.0/10.0 = 0.1 → u/s < r_lo → NEGATIVE
        ds = compute_direction_set(s=100.0, u=1.0, bounds=bounds)
        assert ds.direction == "NEGATIVE"
        assert ds.certified

    def test_unresolved_direction(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        # u/s = 1.0, r_lo = 0.1/10 = 0.01, r_hi = 10/0.1 = 100 → UNRESOLVED
        ds = compute_direction_set(s=1.0, u=1.0, bounds=bounds)
        assert ds.direction == "UNRESOLVED"
        assert not ds.certified


class TestIdentifiability:
    def test_identify_positive(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=1.0)
        result = identify_direction(s=1.0, u=100.0, bounds=bounds)
        assert result.direction == "POSITIVE"
        assert result.certified

    def test_identify_negative(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=1.0, gamma_hi=10.0)
        result = identify_direction(s=100.0, u=1.0, bounds=bounds)
        assert result.direction == "NEGATIVE"
        assert result.certified

    def test_identify_unresolved(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        result = identify_direction(s=1.0, u=1.0, bounds=bounds)
        assert result.direction == "UNRESOLVED"
        assert not result.certified

    def test_counterexample_generation(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        # Observation in unresolved region
        ce = generate_counterexample(s=10.0, u=10.0, bounds=bounds)
        assert ce is not None
        (b1, g1, v1), (b2, g2, v2) = ce
        assert v1 > 0
        assert v2 < 0

    def test_batch_identify(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        obs = np.array([[1.0, 200.0], [200.0, 1.0], [10.0, 10.0]])
        results = batch_identify(obs, bounds)
        assert len(results) == 3
        assert results[0].direction == "POSITIVE"
        assert results[1].direction == "NEGATIVE"
        assert results[2].direction == "UNRESOLVED"

    def test_unresolved_fraction(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        # Generate observations spanning the ratio space
        ratios = np.logspace(-2, 2, 100)
        obs = np.array([[r, 1.0] for r in ratios])  # s=r, u=1 → ratio u/s = 1/r
        result = unresolved_fraction(obs, bounds)
        assert 0 <= result["fraction_unresolved"] <= 1
        assert result["n_total"] == 100


class TestSimulation:
    def test_adversarial_pair_generation(self):
        pair = generate_adversarial_pair(seed=42)
        assert pair is not None
        assert pair.v_a > 0
        assert pair.v_b < 0

    def test_counterexample_family(self):
        family = generate_counterexample_family(n_pairs=10, seed=0)
        assert len(family) > 0
        for p in family:
            assert p.v_a > 0
            assert p.v_b < 0

    def test_simulate_counts(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.3)
        counts = simulate_counts(params, n_cells=100, seed=42)
        assert counts.shape == (100, 2)
        assert np.all(counts >= 0)

    def test_simulate_labeling(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.3)
        labeled = simulate_labeling(params, t_label=0.5, n_cells=100, seed=42)
        assert labeled.shape == (100, 2)
        assert np.all(labeled >= 0)
        assert np.all(labeled <= 1)

    def test_labeling_residual(self):
        params = KineticParams(alpha=1.0, beta=0.5, gamma=0.3)
        # Perfect prediction
        r = labeling_residual(params, 0.5,
                              1.0 - np.exp(-0.3 * 0.5),
                              1.0 - np.exp(-0.5 * 0.5))
        assert r == pytest.approx(0.0, abs=1e-10)
        # Imperfect prediction
        r2 = labeling_residual(params, 0.5, 0.5, 0.5)
        assert r2 > 0


class TestCertificate:
    def test_build_certificate(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=10.0)
        cert = build_certificate(
            gene_id="TEST_GENE",
            dataset_accession="GSE00000",
            dataset_version="v1",
            cell_filter="all",
            s_obs=10.0, u_obs=10.0, n_cells=500,
            bounds=bounds,
        )
        assert cert.gene_id == "TEST_GENE"
        assert cert.direction == "UNRESOLVED"
        assert not cert.certified
        assert cert.certificate_hash is not None

    def test_certificate_summary(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=10.0, gamma_lo=0.1, gamma_hi=1.0)
        cert = build_certificate(
            gene_id="GENE_A",
            dataset_accession="GSE00000",
            dataset_version="v1",
            cell_filter="all",
            s_obs=1.0, u_obs=100.0, n_cells=500,
            bounds=bounds,
        )
        summary = cert.summary()
        assert summary["direction"] == "POSITIVE"
        assert summary["certified"] is True


class TestLabeling:
    def test_labeling_test_runs(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=5.0, gamma_lo=0.1, gamma_hi=5.0)
        result = labeling_falsification_test(
            gene_id="TEST",
            s_obs=10.0, u_obs=10.0, n_cells=200,
            t_label=0.5, bounds=bounds,
            observed_labeled_s=0.3, observed_labeled_u=0.4,
            n_models=50, n_null=100, seed=42,
        )
        assert result.n_compatible_models == 50
        assert 0 <= result.fraction_passing <= 1


class TestValidation:
    def test_synthetic_validation_invariants(self):
        """The three meaningful invariants of the certificate (AUDIT Sec 12):

        - Coverage == 1:  for fixed (s,u), v = βu − γs is LINEAR in (β,γ), so
          the [v_min, v_max] envelope over the parameter rectangle always
          contains the true velocity. If this ever fails, the envelope math
          is wrong.
        - Type-I == 0: a certified sign can never contradict ground truth,
          because certification requires ALL compatible models (including the
          true one) to agree on the sign.
        - Recall == 1 on IND/REP regimes and abstention == 1 on AMBIENT ones.
        """
        bounds = KineticBounds(beta_lo=0.1, beta_hi=5.0, gamma_lo=0.1, gamma_hi=5.0)
        m = validate_on_synthetic(bounds, 150, seed=42)
        assert m.n_genes == 150
        assert m.n_certified > 0, "designed IND/REP regimes must be certifiable"
        assert m.n_unresolved > 0, "designed AMBIENT regimes must abstain"
        assert m.coverage == 1.0, f"linear-envelope identity broken (cov={m.coverage})"
        assert m.type_i_error == 0.0, f"certified sign contradicted truth (={m.type_i_error})"
        assert m.power == 1.0, f"IND/REP recall should be perfect (={m.power})"
        assert m.abstention_rate == 1.0, f"AMBIENT abstention should be perfect"

    def test_counterexample_abstention_is_forced(self):
        """Counterexamples must force abstention: prove opposite-sign models
        fit the snapshot, so the certificate MUST return UNRESOLVED."""
        bounds = KineticBounds(beta_lo=0.1, beta_hi=5.0, gamma_lo=0.1, gamma_hi=5.0)
        result = validate_on_counterexamples(bounds, 100, seed=42)
        assert result["n_pairs"] > 0
        assert result["abstention_rate"] == 1.0, "a counterexample must abstain"

    def test_generic_coverage_and_type1(self):
        """Property test, NOT regime-designed: for ANY (β,γ) inside the bounds
        and any observed (s,u), the envelope must contain the true velocity
        (coverage 1.0) and the certificate can never certify a sign that
        contradicts ground truth (type-I 0.0). Guaranteed by Theorem 1;
        regression-locks the coverage-indentation bug we fixed."""
        bounds = KineticBounds(beta_lo=0.1, beta_hi=5.0, gamma_lo=0.1, gamma_hi=5.0)
        rng = np.random.default_rng(0)
        n_cov = n_t1 = n = 0
        for _ in range(1000):
            beta = float(rng.uniform(bounds.beta_lo, bounds.beta_hi))
            gamma = float(rng.uniform(bounds.gamma_lo, bounds.gamma_hi))
            u = float(rng.uniform(1.0, 100.0))
            s = float(rng.uniform(1.0, 100.0))
            true_v = beta * u - gamma * s
            r = identify_direction(s, u, bounds)
            n += 1
            if r.v_range[0] <= true_v <= r.v_range[1]:
                n_cov += 1
            if r.certified and not (true_v > 0 if r.direction == "POSITIVE"
                                    else true_v < 0):
                n_t1 += 1
        assert n_cov == n, f"envelope missed the true velocity ({n_cov}/{n})"
        assert n_t1 == 0, f"certified a sign contradicting ground truth ({n_t1})"

    def test_labeling_power(self):
        bounds = KineticBounds(beta_lo=0.1, beta_hi=5.0, gamma_lo=0.1, gamma_hi=5.0)
        result = validate_labeling_power(bounds, 20, seed=42)
        assert result["n_genes"] > 0
        assert 0 <= result["power"] <= 1
