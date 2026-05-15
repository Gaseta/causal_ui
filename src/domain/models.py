from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RuntimePaths:
    project_root: Path
    workdir: Path
    data_dir: Path
    outputs_dir: Path
    tables_dir: Path
    figures_dir: Path
    causal_results_dir: Path
    discovery_dir: Path

    @classmethod
    def build(cls, project_root: Path, workdir: Path) -> "RuntimePaths":
        outputs_dir = workdir / "outputs"
        tables_dir = outputs_dir / "tables"
        figures_dir = outputs_dir / "figures"
        causal_results_dir = outputs_dir / "causal_results"
        discovery_dir = outputs_dir / "causal_discovery"
        for p in [workdir, outputs_dir, tables_dir, figures_dir, causal_results_dir, discovery_dir]:
            p.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=project_root,
            workdir=workdir,
            data_dir=project_root / "data",
            outputs_dir=outputs_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            causal_results_dir=causal_results_dir,
            discovery_dir=discovery_dir,
        )


@dataclass(slots=True)
class FileSelection:
    maf_columns: List[str]
    sample_columns: List[str]
    patient_columns: List[str]
    patient_id_col: str
    sample_id_col: str
    maf_gene_col: str
    maf_sample_col: str


@dataclass(slots=True)
class PreprocessingConfig:
    paths: RuntimePaths
    mutations_file: Path
    clinical_sample_file: Path
    clinical_patient_file: Path
    pathway_gmt_file: Optional[Path] = None
    drivers_file: Optional[Path] = None
    pathway_geneset_pickle: Optional[Path] = None
    pathway_networks_pickle: Optional[Path] = None
    selection: Optional[FileSelection] = None
    use_pathways: bool = True
    analyze_subpathways: bool = True
    subgroup_min_fraction: float = 0.10
    fisher_pvalue_threshold: float = 0.05
    fisher_or_threshold: float = 1.0
    discretize_bins: int = 10
    categorical_remap: Dict[str, Dict[str, str]] = field(default_factory=dict)
    discretize_cols_sample: List[str] = field(default_factory=list)
    discretize_cols_patient: List[str] = field(default_factory=list)
    outlier_iqr_factors_sample: Dict[str, float] = field(default_factory=dict)
    outlier_iqr_factors_patient: Dict[str, float] = field(default_factory=dict)
    convert_outcome_to_binary: bool = False
    outcome_positive_regex: str = r"DECEASED|DEAD|DIED|1:DECEASED"
    outcome_binary_columns_sample: List[str] = field(default_factory=list)
    outcome_binary_columns_patient: List[str] = field(default_factory=list)
    convert_stage_to_numeric: bool = False
    stage_roman_regex: str = r"\b(I{1,3}|IV|V)\b"
    stage_roman_columns_sample: List[str] = field(default_factory=list)
    stage_roman_columns_patient: List[str] = field(default_factory=list)
    maf_skiprows_candidates: List[int] = field(default_factory=lambda: [1, 0])
    super_pathways_names: Optional[List[str]] = None
    id_columns: List[str] = field(default_factory=lambda: ["PATIENT_ID", "SAMPLE_ID"])
    row_filters_sample: Dict[str, List[str]] = field(default_factory=dict)
    row_filters_patient: Dict[str, List[str]] = field(default_factory=dict)
    drop_undefined_categorical: bool = True
    pathway_value_strategy: str = "fisher_dense"
    coverage_min_fraction: float = 0.80


@dataclass(slots=True)
class ParameterSchema:
    key: str
    label: str
    kind: str
    default: Any
    help_text: str = ""
    options: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


@dataclass(slots=True)
class AlgorithmSpec:
    key: str
    label: str
    family: str
    backend: str
    runner: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)


@dataclass(slots=True)
class DiscoveryRunConfig:
    dataset_path: Path
    algorithm_key: str
    params: Dict[str, Any]
    paths: RuntimePaths


@dataclass(slots=True)
class InferenceRunConfig:
    dataset_path: Path
    graph_path: Path
    estimate_method: str
    estimate_params: Dict[str, Any]
    refuters: List[str]
    refuter_params: Dict[str, Dict[str, Any]]
    treatment_columns: List[str]
    outcome_columns: List[str]
    all_pairs: bool
    paths: RuntimePaths