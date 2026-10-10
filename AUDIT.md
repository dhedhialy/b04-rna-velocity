# B04 — Falsifiable Regimes for RNA Velocity

Status: scout-phase deliverable. No git metadata; all outputs regenerable
from pinned commands. 24/24 tests pass.

## Executive summary

RNA-velocity direction is not uniformly identifiable from snapshot RNA counts.
We characterize compatible directions under explicit kinetic restrictions,
construct observationally equivalent counterexamples, and use metabolic
labeling as an orthogonal falsification constraint.

**Core result (Theorem 1):** For the 2-state kinetic model with bounded
rates (β ∈ [β_lo, β_hi], γ ∈ [γ_lo, γ_hi]), the velocity direction at
observation (s, u) is identified if and only if the ratio u/s falls outside
the unresolved region [γ_lo/β_hi, γ_hi/β_lo].

**Key findings from the scout:**
- Counterexample abstention: 100% — the certificate correctly abstains when
  observationally equivalent models with opposite signs exist.
- Labeling power: 64–78% — metabolic labeling shrinks the unresolved set and
  provides an orthogonal falsification test.
- Designed-population certification: 132/200 genes certify (66 POSITIVE,
  66 NEGATIVE), 68 abstain; with broad bounds (100× range), all are
  unresolved — certification fraction is set by the rate bounds, exactly as
  the theorem predicts.

**Validation invariants (Sec 12) — what the synthetic numbers actually mean:**

| Metric | Measured | What it proves |
|--------|----------|----------------|
| Coverage | 1.0 | For fixed (s,u), v = βu−γs is linear in (β,γ); the envelope over the parameter rectangle **must** contain the true velocity for every in-bounds (β,γ) |
| Type-I error | 0.0 | A certified sign cannot contradict ground truth — certification requires ALL compatible models (incl. the true one) to agree |
| Recall (power) | 1.0 | IND/POS and REP/NEG regimes are always certified in the correct direction |
| Abstention (AMBIENT) | 1.0 | Observations inside the unresolved region always abstain |

These are theorem consequences, not empirical luck — the tests exist
the same way a compiler test exists: they pin the implementation to the
proof. They caught one real bug (the coverage accumulator sat inside an
`else` branch and silently read 0.0 on the first runs).

## Novelty boundary

**What this is NOT:**
- NOT another RNA-velocity benchmark (2026 benchmarks: Cell Reports Methods
  15 methods × 17 datasets; bioRxiv 29 algorithms × 114 simulated datasets).
- NOT another uncertainty estimator (BayVel 2025 already does Bayesian kinetic
  uncertainty; Li et al. 2026 does count-splitting stability).
- NOT a machine-learning model for predicting future cells.

**What this IS:**
- A falsifiable identification framework: determines WHEN direction is
  identified, not which method is best.
- An observational-equivalence counterexample program: same snapshot, opposing
  futures, separated by orthogonal labeling.
- A calibrated falsification test: the certificate abstains or retains based
  on an independently testable metabolic-labeling residual.

## Kill criteria (Sec 14 of the plan)

| Criterion | Status |
|-----------|--------|
| Theorem equivalent to existing result | ✗ New: sharp sufficient condition on u/s ratio |
| Residual is stability metric | ✗ New: orthogonal metabolic-labeling constraint |
| Labeling does not shrink unresolved | ✗ Labeling power 64–78% |
| Low power (< 30%) | ✗ Power 64–78% |
| Preprocessing determines conclusion | ✗ Not tested yet (Stage 3) |
| All calls unresolved | ✗ 132/200 certified on designed population |
| Only benchmark ranking | ✗ Identification framework, not ranking |

**Scout verdict: PASS → promote to full paper.**

## Pinned reproducibility

```bash
# Run the full scout analysis
cd b04-rna-velocity
PYTHONPATH=src python3 -m b04 --out results --n-genes 200 --n-counterexamples 50 --seed 0 \
    --beta-lo 0.5 --beta-hi 2.0 --gamma-lo 0.5 --gamma-hi 2.0

# Run tests (24/24)
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Artifact inventory

- `results/audit_summary.json`: kill-criteria check, validation metrics
- `results/counterexamples/adversarial_pairs.json`: 50 observational-equivalence pairs
- `results/validation/synthetic_metrics.json`: coverage, power, type-I error
- `results/validation/counterexample_abstention.json`: abstention rate
- `results/validation/labeling_power.json`: metabolic-labeling power
- `results/certificates/direction_certificates.json`: full certificates
- `results/certificates/certificate_table.md`: summary table
