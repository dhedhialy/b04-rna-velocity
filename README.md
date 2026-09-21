# B04 — Falsifiable Regimes for RNA Velocity

A falsifiable identification framework for RNA-velocity direction under
explicit kinetic assumptions, with orthogonal metabolic-labeling falsification.

## What this does

RNA velocity attempts to infer transcriptional direction from snapshot
spliced/unspliced RNA counts. But under the 2-state kinetic model, the
velocity v = βu − γs depends on unknown rates β, γ. Two models with
different rates can produce the same snapshot but opposite velocity signs.

This framework:
1. **Characterizes compatible directions** — computes the velocity range
   [v_min, v_max] over all kinetic parameters consistent with the data.
2. **Generates counterexamples** — constructs explicit pairs of models that
   are observationally equivalent under snapshots but have opposite signs.
3. **Issues direction certificates** — POSITIVE (all compatible models agree
   v > 0), NEGATIVE (all agree v < 0), or UNRESOLVED (models disagree).
4. **Tests with metabolic labeling** — uses orthogonal labeling data to
   falsify snapshot-compatible models that fail to predict held-out
   observations.

## Quick start

```bash
pip install -e .
PYTHONPATH=src python3 -m b04 --out results
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Structure

```
src/b04/
  kinetic.py          2-state model, direction sets
  identifiability.py  Theorem 1, counterexamples
  simulation.py       synthetic data, adversarial pairs
  certificate.py      direction certificate schema
  labeling.py         metabolic-labeling falsification test
  validation.py       type-I error, coverage, power
  reporting.py        audit summary generation
```

## Key theorem

**Theorem 1 (Direction identification).** Under the 2-state model with
bounded rates β ∈ [β_lo, β_hi], γ ∈ [γ_lo, γ_hi]:

- If u/s > γ_hi/β_lo → POSITIVE (gene is being induced)
- If u/s < γ_lo/β_hi → NEGATIVE (gene is being repressed)
- Otherwise → UNRESOLVED (compatible models give both signs)

The width of the unresolved region is W = γ_hi/β_lo − γ_lo/β_hi.
Identification improves as rate bounds tighten.

## References

- Bergen et al. (2020), scVelo, Nature Biotechnology
- La Manno et al. (2018), RNA velocity, Nature
- Sabbioni et al. (2025), BayVel, arXiv:2505.03083
- Li et al. (2026), count-splitting stability, Bioinformatics Advances
- Cell Reports Methods (2026), 15-method benchmark
- Huang et al. (2026), 29-algorithm benchmark, bioRxiv
