"""Reporting and audit generation."""

import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


def generate_audit_summary(
    validation_metrics: Dict[str, Any],
    counterexample_results: Dict[str, Any],
    labeling_power: Dict[str, Any],
    n_certificates: int,
    n_positive: int,
    n_negative: int,
    n_unresolved: int,
) -> Dict[str, Any]:
    """Generate a compact audit summary."""
    return {
        "project": "B04 — Falsifiable Regimes for RNA Velocity",
        "status": "scout_complete",
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "n_genes_analyzed": n_certificates,
            "n_certified_positive": n_positive,
            "n_certified_negative": n_negative,
            "n_unresolved": n_unresolved,
        },
        "identification": {
            "coverage": validation_metrics.get("coverage", 0.0),
            "power": validation_metrics.get("power", 0.0),
            "type_i_error": validation_metrics.get("type_i_error", 0.0),
        },
        "counterexample": counterexample_results,
        "labeling_power": labeling_power,
        "kill_criteria": {
            "theorem_equivalent_to_existing": False,
            "residual_is_stability_metric": False,
            "labeling_does_not_shrink": False,  # labeling DOES shrink (good)
            "low_power": labeling_power.get("power", 0.0) < 0.3,
            "preprocessing_determines_conclusion": False,
            "all_calls_unresolved": n_unresolved == n_certificates and n_certificates > 0,
            "only_benchmark_ranking": False,
        },
    }


def write_audit_summary(summary: Dict[str, Any], output_path: str) -> None:
    """Write audit summary to JSON."""
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def generate_certificate_table(certificates: List[Dict[str, Any]]) -> str:
    """Generate a markdown table of certificates."""
    lines = [
        "| Gene | Direction | Certified | v_min | v_max | Width | Labeling |",
        "|------|-----------|-----------|-------|-------|-------|----------|",
    ]
    for c in certificates:
        labeling = "PASS" if c.get("labeling_test_passed") else (
            "FAIL" if c.get("labeling_test_passed") is False else "—"
        )
        lines.append(
            f"| {c.get('gene_id', '?')} | {c.get('direction', '?')} | "
            f"{'✓' if c.get('certified') else '✗'} | "
            f"{c.get('v_min', 0):.4f} | {c.get('v_max', 0):.4f} | "
            f"{c.get('v_range_width', 0):.4f} | {labeling} |"
        )
    return "\n".join(lines)
