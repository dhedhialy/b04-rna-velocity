"""B04 main entry point: run the full scout analysis."""

import argparse
import json
import numpy as np
from pathlib import Path

from .kinetic import KineticBounds
from .identifiability import batch_identify, unresolved_fraction, generate_counterexample
from .simulation import generate_counterexample_family, simulate_counts, simulate_labeling
from .certificate import build_certificate, export_certificates, certificate_summary_table
from .labeling import labeling_falsification_test
from .validation import (
    validate_on_synthetic, validate_on_counterexamples,
    validate_labeling_power, synthetic_population,
)
from .reporting import generate_audit_summary, write_audit_summary


def main():
    ap = argparse.ArgumentParser(description="B04 — Falsifiable Regimes for RNA Velocity")
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help="output directory")
    ap.add_argument("--n-genes", type=int, default=200,
                    help="number of synthetic genes for validation")
    ap.add_argument("--n-counterexamples", type=int, default=50,
                    help="number of adversarial pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-lo", type=float, default=0.05)
    ap.add_argument("--beta-hi", type=float, default=5.0)
    ap.add_argument("--gamma-lo", type=float, default=0.05)
    ap.add_argument("--gamma-hi", type=float, default=5.0)
    ap.add_argument("--t-label", type=float, default=0.5,
                    help="metabolic-labeling time")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "certificates").mkdir(exist_ok=True)
    (args.out / "counterexamples").mkdir(exist_ok=True)
    (args.out / "validation").mkdir(exist_ok=True)

    bounds = KineticBounds(
        beta_lo=args.beta_lo, beta_hi=args.beta_hi,
        gamma_lo=args.gamma_lo, gamma_hi=args.gamma_hi,
    )

    rng = np.random.default_rng(args.seed)

    # --- Stage 0: Observational-equivalence counterexamples ---
    print("=== Stage 0: Counterexample generation ===")
    ce_family = generate_counterexample_family(
        n_pairs=args.n_counterexamples, seed=args.seed,
    )
    print(f"  Generated {len(ce_family)} adversarial pairs")

    # Save counterexamples
    ce_data = []
    for i, p in enumerate(ce_family):
        ce_data.append({
            "pair_id": i,
            "s": round(p.s, 4), "u": round(p.u, 4),
            "v_a": round(p.v_a, 4), "v_b": round(p.v_b, 4),
            "theta_a": {"alpha": round(p.theta_a.alpha, 4),
                        "beta": round(p.theta_a.beta, 4),
                        "gamma": round(p.theta_a.gamma, 4)},
            "theta_b": {"alpha": round(p.theta_b.alpha, 4),
                        "beta": round(p.theta_b.beta, 4),
                        "gamma": round(p.theta_b.gamma, 4)},
        })
    with open(args.out / "counterexamples" / "adversarial_pairs.json", 'w') as f:
        json.dump(ce_data, f, indent=2)

    # --- Stage 1: Synthetic validation ---
    print("\n=== Stage 1: Synthetic validation ===")
    val_metrics = validate_on_synthetic(bounds, args.n_genes, seed=args.seed)
    print(f"  Coverage: {val_metrics.coverage:.3f}")
    print(f"  Power: {val_metrics.power:.3f}")
    print(f"  Type-I error: {val_metrics.type_i_error:.3f}")
    print(f"  Abstention rate: {val_metrics.abstention_rate:.3f}")

    # Save validation metrics
    val_dict = {
        "n_genes": val_metrics.n_genes,
        "n_certified": val_metrics.n_certified,
        "n_unresolved": val_metrics.n_unresolved,
        "coverage": round(val_metrics.coverage, 4),
        "power": round(val_metrics.power, 4),
        "type_i_error": round(val_metrics.type_i_error, 4),
        "abstention_rate": round(val_metrics.abstention_rate, 4),
        "n_false_positive": val_metrics.n_false_positive,
        "n_false_negative": val_metrics.n_false_negative,
    }
    with open(args.out / "validation" / "synthetic_metrics.json", 'w') as f:
        json.dump(val_dict, f, indent=2)

    # --- Stage 2: Counterexample abstention ---
    print("\n=== Stage 2: Counterexample abstention ===")
    ce_val = validate_on_counterexamples(bounds, args.n_counterexamples, seed=args.seed)
    print(f"  Abstention rate on counterexamples: {ce_val['abstention_rate']:.3f}")

    with open(args.out / "validation" / "counterexample_abstention.json", 'w') as f:
        json.dump(ce_val, f, indent=2)

    # --- Stage 3: Labeling power ---
    print("\n=== Stage 3: Labeling power ===")
    labeling_pwr = validate_labeling_power(bounds, args.n_genes, args.t_label, seed=args.seed)
    print(f"  Labeling power: {labeling_pwr['power']:.3f}")

    with open(args.out / "validation" / "labeling_power.json", 'w') as f:
        json.dump(labeling_pwr, f, indent=2)

    # --- Stage 4: Certificates for the same designed validation population ---
    print("\n=== Stage 4: Certificate generation (designed genes with ground truth) ===")
    certificates = []
    n_pos, n_neg, n_unr = 0, 0, 0
    rows = synthetic_population(bounds, args.n_genes, seed=args.seed)
    for row in rows:
        cert = build_certificate(
            gene_id=row["gene_id"],
            dataset_accession="synthetic",
            dataset_version="v0",
            cell_filter="all",
            s_obs=row["s"], u_obs=row["u"], n_cells=200,
            bounds=bounds,
        )
        certificates.append(cert)
        if cert.direction == "POSITIVE":
            n_pos += 1
        elif cert.direction == "NEGATIVE":
            n_neg += 1
        else:
            n_unr += 1

    export_certificates(certificates,
                        str(args.out / "certificates" / "direction_certificates.json"))

    # Save summary table
    with open(args.out / "certificates" / "certificate_table.md", 'w') as f:
        f.write(certificate_summary_table(certificates))

    print(f"  Certificates: {n_pos} POSITIVE, {n_neg} NEGATIVE, {n_unr} UNRESOLVED")

    # --- Audit summary ---
    print("\n=== Audit summary ===")
    audit = generate_audit_summary(
        validation_metrics=val_dict,
        counterexample_results=ce_val,
        labeling_power=labeling_pwr,
        n_certificates=len(certificates),
        n_positive=n_pos, n_negative=n_neg, n_unresolved=n_unr,
    )
    write_audit_summary(audit, str(args.out / "audit_summary.json"))

    # Print summary
    print(f"\nSummary: {len(certificates)} genes analyzed")
    print(f"  Certified: {n_pos + n_neg} ({(n_pos+n_neg)/len(certificates)*100:.1f}%)")
    print(f"  Unresolved: {n_unr} ({n_unr/len(certificates)*100:.1f}%)")
    print(f"  Coverage: {val_dict['coverage']}")
    print(f"  Power: {val_dict['power']}")
    print(f"  Type-I error: {val_dict['type_i_error']}")
    print(f"  Counterexample abstention: {ce_val['abstention_rate']}")
    print(f"  Labeling power: {labeling_pwr['power']}")

    # Kill criteria check
    kill = audit["kill_criteria"]
    kill_reasons = [k for k, v in kill.items() if v]
    if kill_reasons:
        print(f"\n⚠ KILL CRITERIA TRIGGERED: {', '.join(kill_reasons)}")
        print("  → Project should be killed, merged, or reframed.")
    else:
        print("\n✓ No kill criteria triggered. Scout passes → promote to full paper.")

    print(f"\nResults written to {args.out}/")


if __name__ == "__main__":
    main()
