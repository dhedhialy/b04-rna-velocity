"""Direction certificate schema (AUDIT Sec 8).

A certificate carries: dataset version, inclusion rules, estimand,
nuisance set, fitted parameters, interval/direction set, optimization
evidence, held-out fold, negative-control result, and semantic status.

The certificate is immutable once produced. The agent may propose
analysis, but should not silently mutate the estimand or validation split.
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from .kinetic import KineticBounds, DirectionSet
from .identifiability import IdentificationResult


@dataclass(frozen=True)
class Certificate:
    """Immutable direction certificate for a single gene.

    The certificate captures:
    - Dataset and gene identity
    - Kinetic bounds (the nuisance set)
    - Direction set computation (v_min, v_max, direction)
    - Identification status (POSITIVE, NEGATIVE, UNRESOLVED)
    - Optimization evidence (method, convergence)
    - Validation status (held-out labeling test, if available)
    - Negative control result (if available)
    """
    # Identity
    gene_id: str
    dataset_accession: str
    dataset_version: str
    cell_filter: str

    # Estimand
    estimand: str = "velocity_direction"

    # Kinetic bounds (nuisance set)
    beta_lo: float = 0.01
    beta_hi: float = 10.0
    gamma_lo: float = 0.01
    gamma_hi: float = 10.0
    alpha_lo: Optional[float] = None
    alpha_hi: Optional[float] = None
    ratio_lo: Optional[float] = None
    ratio_hi: Optional[float] = None

    # Observation
    s_obs: float = 0.0       # mean spliced count
    u_obs: float = 0.0       # mean unspliced count
    n_cells: int = 0

    # Direction set
    v_min: float = 0.0
    v_max: float = 0.0

    # Identification status
    direction: str = "UNRESOLVED"  # POSITIVE, NEGATIVE, UNRESOLVED

    # Optimization evidence
    optimization_method: str = "corner_evaluation"
    optimization_converged: bool = True
    optimization_certificate: Optional[str] = None

    # Validation (held-out labeling test)
    labeling_test_performed: bool = False
    labeling_test_passed: Optional[bool] = None
    labeling_residual: Optional[float] = None
    labeling_p_value: Optional[float] = None

    # Negative control
    negative_control_performed: bool = False
    negative_control_result: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    assumptions: List[str] = field(default_factory=lambda: [
        "A1: constant (α, β, γ) within the observation window",
        "A2: bounded rates in [β_lo, β_hi] × [γ_lo, γ_hi]",
        "A3: Poisson observation model for counts",
    ])

    @property
    def v_range_width(self) -> float:
        return self.v_max - self.v_min

    @property
    def certified(self) -> bool:
        return self.direction in ("POSITIVE", "NEGATIVE")

    @property
    def certificate_hash(self) -> str:
        """Immutable hash of the certificate (for integrity checks)."""
        d = asdict(self)
        d.pop('created_at', None)  # created_at is not part of the hash
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        """Compact summary for audit/reproducibility."""
        return {
            "gene_id": self.gene_id,
            "dataset": f"{self.dataset_accession} ({self.dataset_version})",
            "direction": self.direction,
            "certified": self.certified,
            "v_range": [round(self.v_min, 4), round(self.v_max, 4)],
            "v_range_width": round(self.v_range_width, 4),
            "ratio_su": round(self.u_obs / self.s_obs, 4) if self.s_obs > 0 else None,
            "assumptions": self.assumptions,
            "labeling_test": self.labeling_test_passed,
            "hash": self.certificate_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization (for JSON export)."""
        return asdict(self)


def build_certificate(
    gene_id: str,
    dataset_accession: str,
    dataset_version: str,
    cell_filter: str,
    s_obs: float,
    u_obs: float,
    n_cells: int,
    bounds: KineticBounds,
    id_result: Optional[IdentificationResult] = None,
    labeling_test_passed: Optional[bool] = None,
    labeling_residual: Optional[float] = None,
    labeling_p_value: Optional[float] = None,
    negative_control_result: Optional[str] = None,
) -> Certificate:
    """Build a direction certificate from observation and identification results."""
    if id_result is None:
        from .identifiability import identify_direction
        id_result = identify_direction(s_obs, u_obs, bounds)

    return Certificate(
        gene_id=gene_id,
        dataset_accession=dataset_accession,
        dataset_version=dataset_version,
        cell_filter=cell_filter,
        beta_lo=bounds.beta_lo, beta_hi=bounds.beta_hi,
        gamma_lo=bounds.gamma_lo, gamma_hi=bounds.gamma_hi,
        alpha_lo=bounds.alpha_lo, alpha_hi=bounds.alpha_hi,
        ratio_lo=bounds.ratio_lo, ratio_hi=bounds.ratio_hi,
        s_obs=s_obs, u_obs=u_obs, n_cells=n_cells,
        v_min=id_result.v_range[0], v_max=id_result.v_range[1],
        direction=id_result.direction,
        labeling_test_performed=labeling_test_passed is not None,
        labeling_test_passed=labeling_test_passed,
        labeling_residual=labeling_residual,
        labeling_p_value=labeling_p_value,
        negative_control_performed=negative_control_result is not None,
        negative_control_result=negative_control_result,
    )


def export_certificates(
    certificates: List[Certificate],
    output_path: str,
) -> None:
    """Export a list of certificates to JSON."""
    data = [c.to_dict() for c in certificates]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def certificate_summary_table(certificates: List[Certificate]) -> str:
    """Generate a markdown summary table of certificates."""
    lines = [
        "| Gene | Dataset | Direction | Certified | v_min | v_max | Width | Labeling |",
        "|------|---------|-----------|-----------|-------|-------|-------|----------|",
    ]
    for c in certificates:
        labeling = "PASS" if c.labeling_test_passed else ("FAIL" if c.labeling_test_passed is False else "—")
        lines.append(
            f"| {c.gene_id} | {c.dataset_accession} | {c.direction} | "
            f"{'✓' if c.certified else '✗'} | {c.v_min:.4f} | {c.v_max:.4f} | "
            f"{c.v_range_width:.4f} | {labeling} |"
        )
    return "\n".join(lines)
