from __future__ import annotations

import pickle
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.domain.models import PreprocessingConfig
from src.services.io_utils import ensure_required_columns, normalize_columns, read_table, safe_filename


DEFAULT_SUPER_PATHWAYS = [
    "Autophagy",
    "Cell Cycle",
    "Cell-Cell communication",
    "Cellular responses to stimuli",
    "Chromatin organization",
    "Developmental Biology",
    "DNA Repair",
    "DNA Replication",
    "Extracellular matrix organization",
    "Gene expression (Transcription)",
    "Immune System",
    "Metabolism",
    "Metabolism of proteins",
    "Neuronal System",
    "Programmed Cell Death",
    "Signal Transduction",
    "Transport of small molecules",
    "Vesicle-mediated transport",
]


def _log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, float) and np.isnan(value):
        return pd.NA
    text = str(value).strip()
    if text in {"", "NA", "N/A", "<NA>", "nan", "None"}:
        return pd.NA
    return text


def _normalize_gene_symbol(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().upper()


def discretize_in_groups(values: pd.Series, number_of_bins: int = 10) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.dropna().empty:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="string")
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx == mn:
        label = f"{round(mn)}-{round(mx)}"
        return pd.Series([label] * len(values), index=values.index, dtype="string")
    bin_range = (mx - mn) / float(number_of_bins)
    bins = [mn]
    for i in range(number_of_bins):
        bins.append(mn + bin_range * (i + 1))
    labels = [f"{round(bins[i])}-{round(bins[i + 1])}" for i in range(len(bins) - 1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype("string")


def _iqr_upper_threshold(values: pd.Series, factor: float) -> Optional[float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return None
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    return float(q3 + factor * (q3 - q1))


def _roman_to_int(roman: str) -> Optional[int]:
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    return mapping.get(str(roman).upper().strip())


def _apply_categorical_remap(df: pd.DataFrame, remap: Dict[str, Dict[str, str]], label: str) -> None:
    for col, mp in (remap or {}).items():
        if col in df.columns and isinstance(mp, dict) and mp:
            df[col] = df[col].replace(mp)
            _log(f"{label}: remapped {col}")


def _apply_outliers_iqr(df: pd.DataFrame, factors: Dict[str, float], label: str) -> pd.DataFrame:
    out = df
    for col, factor in (factors or {}).items():
        if col not in out.columns:
            continue
        thr = _iqr_upper_threshold(out[col], float(factor))
        if thr is None:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        before = len(out)
        out = out[numeric < thr].copy()
        _log(f"{label}: removed outliers for {col} ({before} -> {len(out)})")
    return out


def _apply_discretization(df: pd.DataFrame, cols: List[str], bins: int, label: str) -> None:
    for col in cols or []:
        if col in df.columns:
            df[f"{col}__discrete"] = discretize_in_groups(df[col], bins)
            _log(f"{label}: discretized {col}")


def _apply_outcome_binary(df: pd.DataFrame, cols: List[str], positive_regex: str, label: str) -> None:
    rx = re.compile(str(positive_regex), flags=re.IGNORECASE)
    for col in cols or []:
        if col in df.columns:
            s = df[col].astype("string")
            df[f"{col}__binary"] = s.apply(
                lambda v: 1 if (v is not pd.NA and rx.search(str(v) or "") is not None) else 0
            ).astype("Int64")
            _log(f"{label}: binary outcome {col}")


def _apply_stage_roman(df: pd.DataFrame, cols: List[str], roman_regex: str, label: str) -> None:
    rx = re.compile(str(roman_regex), flags=re.IGNORECASE)

    def _conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        m = rx.search(str(v))
        return _roman_to_int(m.group(1)) if m else pd.NA

    for col in cols or []:
        if col in df.columns:
            df[f"{col}__num"] = df[col].apply(_conv).astype("Int64")
            _log(f"{label}: roman stage {col}")


def _parse_gmt(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathway_name = str(parts[0]).strip()
                genes = sorted({_normalize_gene_symbol(g) for g in parts[2:] if _normalize_gene_symbol(g)})
                if pathway_name and genes:
                    out[pathway_name] = genes
    return out


def read_driver_genes(cfg: PreprocessingConfig) -> set[str]:
    if cfg.drivers_file and cfg.drivers_file.exists():
        df = pd.read_csv(cfg.drivers_file, sep="\t", low_memory=False)
        gene_col = "symbol" if "symbol" in df.columns else df.columns[0]
        drivers = {
            _normalize_gene_symbol(v)
            for v in df[gene_col].dropna().astype(str).tolist()
            if _normalize_gene_symbol(v)
        }
        _log(f"Loaded drivers={len(drivers)} from {cfg.drivers_file}")
        return drivers
    _log("No drivers file provided. Using all genes from MAF as eligible drivers.")
    return set()


def load_pathway_hierarchy(cfg: PreprocessingConfig) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    geneset_pkl = cfg.pathway_geneset_pickle
    networks_pkl = cfg.pathway_networks_pickle
    if geneset_pkl and networks_pkl and geneset_pkl.exists() and networks_pkl.exists():
        with open(geneset_pkl, "rb") as f:
            pathway_gene_set = pickle.load(f)
        with open(networks_pkl, "rb") as f:
            pathway_networks = pickle.load(f)
        pathway_gene_set = {
            str(name).strip(): sorted({_normalize_gene_symbol(g) for g in genes if _normalize_gene_symbol(g)})
            for name, genes in dict(pathway_gene_set).items()
            if str(name).strip()
        }
        pathway_networks = {str(name).strip(): graph for name, graph in dict(pathway_networks).items() if str(name).strip()}
        return pathway_gene_set, pathway_networks
    return {}, {}


def significant_intersection(neither: int, anotb: int, bnota: int, both: int) -> Dict[str, float]:
    oddsratio, p_fisher = stats.fisher_exact([[both, anotb], [bnota, neither]], alternative="two-sided")
    return {
        "OR": float(oddsratio),
        "pValue_raw": float(p_fisher),
        "pValue": round(float(p_fisher), 6),
    }


def _bfs_nodes(graph: Any, start: str) -> List[str]:
    visited: Set[str] = {start}
    out: List[str] = []
    q = deque([start])
    while q:
        u = q.popleft()
        out.append(u)
        try:
            neigh = list(graph.neighbors(u))
        except Exception:
            neigh = []
        for v in neigh:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return out


def _deduplicate_by_first(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if key_col not in df.columns:
        return df
    return df.dropna(subset=[key_col]).drop_duplicates(subset=[key_col], keep="first").copy()


def _series_to_clean_dict(row: pd.Series, cols: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        out[c] = _clean_scalar(row[c]) if c in row.index else pd.NA
    return out


def _build_pathway_catalog(
    pathway_gene_set: Dict[str, List[str]],
    pathway_networks: Dict[str, Any],
    include_subpathways: bool,
    selected_super_pathways: Optional[List[str]],
) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    networks_available = bool(pathway_networks)
    roots = selected_super_pathways or list(pathway_networks.keys()) or DEFAULT_SUPER_PATHWAYS

    if networks_available:
        for root in roots:
            if root not in pathway_networks:
                continue
            for node in _bfs_nodes(pathway_networks[root], root):
                if node in seen:
                    continue
                if node == root:
                    level = "super"
                else:
                    if not include_subpathways:
                        continue
                    level = "sub"
                if node in pathway_gene_set:
                    seen.add(node)
                    catalog.append(
                        {
                            "pathway": node,
                            "root": root,
                            "level": level,
                            "output_name": node if node == root else f"{root} -- {node}",
                            "genes": pathway_gene_set[node],
                        }
                    )

    for pathway_name, genes in pathway_gene_set.items():
        if pathway_name in seen:
            continue
        seen.add(pathway_name)
        catalog.append(
            {
                "pathway": pathway_name,
                "root": pathway_name,
                "level": "standalone",
                "output_name": pathway_name,
                "genes": genes,
            }
        )

    catalog.sort(key=lambda x: (str(x["root"]), 0 if x["level"] == "super" else 1, str(x["pathway"])))
    return catalog


def prepare_data(cfg: PreprocessingConfig) -> dict[str, Any]:
    if not cfg.selection:
        raise ValueError("PreprocessingConfig.selection is required.")

    selection = cfg.selection
    datasets_dir = cfg.paths.tables_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    maf = None
    for sk in cfg.maf_skiprows_candidates:
        try:
            maf = read_table(cfg.mutations_file, usecols=selection.maf_columns, skiprows=int(sk))
            if maf.shape[1] > 1:
                break
        except Exception:
            continue
    if maf is None:
        raise ValueError("Could not read MAF file.")

    sample = read_table(cfg.clinical_sample_file, usecols=selection.sample_columns)
    patient = read_table(cfg.clinical_patient_file, usecols=selection.patient_columns)
    maf = normalize_columns(maf)
    sample = normalize_columns(sample)
    patient = normalize_columns(patient)

    ensure_required_columns(maf, [selection.maf_gene_col, selection.maf_sample_col], "MAF")
    ensure_required_columns(sample, [selection.patient_id_col, selection.sample_id_col], "SAMPLE")
    ensure_required_columns(patient, [selection.patient_id_col], "PATIENT")

    maf[selection.maf_gene_col] = maf[selection.maf_gene_col].map(_normalize_gene_symbol)
    maf[selection.maf_sample_col] = maf[selection.maf_sample_col].map(_clean_scalar).astype("string")
    sample[selection.patient_id_col] = sample[selection.patient_id_col].map(_clean_scalar).astype("string")
    sample[selection.sample_id_col] = sample[selection.sample_id_col].map(_clean_scalar).astype("string")
    patient[selection.patient_id_col] = patient[selection.patient_id_col].map(_clean_scalar).astype("string")

    maf = maf[(maf[selection.maf_gene_col] != "") & (maf[selection.maf_sample_col].notna())].copy()
    sample = sample[sample[selection.patient_id_col].notna() & sample[selection.sample_id_col].notna()].copy()
    patient = patient[patient[selection.patient_id_col].notna()].copy()

    _apply_categorical_remap(sample, cfg.categorical_remap, "SAMPLE")
    _apply_categorical_remap(patient, cfg.categorical_remap, "PATIENT")
    sample = _apply_outliers_iqr(sample, cfg.outlier_iqr_factors_sample, "SAMPLE")
    patient = _apply_outliers_iqr(patient, cfg.outlier_iqr_factors_patient, "PATIENT")
    _apply_discretization(sample, cfg.discretize_cols_sample, cfg.discretize_bins, "SAMPLE")
    _apply_discretization(patient, cfg.discretize_cols_patient, cfg.discretize_bins, "PATIENT")
    if cfg.convert_outcome_to_binary:
        _apply_outcome_binary(sample, cfg.outcome_binary_columns_sample, cfg.outcome_positive_regex, "SAMPLE")
        _apply_outcome_binary(patient, cfg.outcome_binary_columns_patient, cfg.outcome_positive_regex, "PATIENT")
    if cfg.convert_stage_to_numeric:
        _apply_stage_roman(sample, cfg.stage_roman_columns_sample, cfg.stage_roman_regex, "SAMPLE")
        _apply_stage_roman(patient, cfg.stage_roman_columns_patient, cfg.stage_roman_regex, "PATIENT")

    sample = _deduplicate_by_first(sample, selection.sample_id_col)
    patient = _deduplicate_by_first(patient, selection.patient_id_col)

    valid_patients = set(patient[selection.patient_id_col].dropna().astype(str))
    sample = sample[sample[selection.patient_id_col].isin(valid_patients)].copy()
    valid_samples = set(sample[selection.sample_id_col].dropna().astype(str))
    maf = maf[maf[selection.maf_sample_col].isin(valid_samples)].copy()

    drivers = read_driver_genes(cfg)
    if drivers:
        maf_filtered = maf[maf[selection.maf_gene_col].isin(drivers)].copy()
    else:
        maf_filtered = maf.copy()

    maf_filtered = maf_filtered[[selection.maf_sample_col, selection.maf_gene_col]].dropna().drop_duplicates().copy()
    universe_genes: Set[str] = set(maf_filtered[selection.maf_gene_col].astype(str).unique())

    pathway_gene_set, pathway_networks = load_pathway_hierarchy(cfg)
    gmt_gene_sets: Dict[str, List[str]] = {}
    if cfg.use_pathways and cfg.pathway_gmt_file and cfg.pathway_gmt_file.exists():
        gmt_gene_sets = _parse_gmt(cfg.pathway_gmt_file)

    if not pathway_gene_set:
        pathway_gene_set = gmt_gene_sets
    elif gmt_gene_sets:
        for pathway_name, genes in gmt_gene_sets.items():
            pathway_gene_set.setdefault(pathway_name, genes)

    if not cfg.use_pathways or not pathway_gene_set:
        raise ValueError("No pathway definitions were found. Provide pathwayGeneSet/pathwayNetworks and/or ReactomePathways.gmt.")

    pathway_catalog = _build_pathway_catalog(
        pathway_gene_set=pathway_gene_set,
        pathway_networks=pathway_networks,
        include_subpathways=cfg.analyze_subpathways,
        selected_super_pathways=cfg.super_pathways_names,
    )

    number_of_patients = int(sample[selection.patient_id_col].nunique())
    min_patients = max(1, int(round(number_of_patients * cfg.subgroup_min_fraction)))

    sample_derived = [c for c in sample.columns if c.endswith("__discrete") or c.endswith("__binary") or c.endswith("__num")]
    patient_derived = [c for c in patient.columns if c.endswith("__discrete") or c.endswith("__binary") or c.endswith("__num")]
    out_sample_cols = [
        c for c in selection.sample_columns if c in sample.columns and c not in {selection.patient_id_col, selection.sample_id_col}
    ] + sample_derived
    out_patient_cols = [c for c in selection.patient_columns if c in patient.columns and c != selection.patient_id_col] + patient_derived

    sample_by_id = sample.set_index(selection.sample_id_col, drop=False)
    patient_by_id = patient.set_index(selection.patient_id_col, drop=False)

    sample_gene_map: Dict[str, Set[str]] = {
        str(sample_id): set(df_grp[selection.maf_gene_col].astype(str).tolist())
        for sample_id, df_grp in maf_filtered.groupby(selection.maf_sample_col)
    }

    group_resume_records: List[Dict[str, Any]] = []
    patient_vs_pathways: Dict[str, Dict[str, int]] = {}
    datasets_written: List[Path] = []
    datasets: Dict[str, pd.DataFrame] = {}

    def _evaluate_pathway(pathway_info: Dict[str, Any]) -> None:
        output_name = str(pathway_info["output_name"])
        pathway_name = str(pathway_info["pathway"])
        root_name = str(pathway_info["root"])
        level = str(pathway_info["level"])
        genes_pathway = set(pathway_info.get("genes") or []) & universe_genes
        if not genes_pathway:
            return

        patient_vs_pathways[output_name] = {}
        df_out: Dict[str, List[object]] = {
            "PatientID": [],
            "SampleID": [],
            "PathwayRoot": [],
            "PathwayName": [],
            "PathwayLevel": [],
        }
        for c in out_sample_cols:
            df_out[f"SAMPLE__{c}"] = []
        for c in out_patient_cols:
            df_out[f"PATIENT__{c}"] = []
        df_out[output_name] = []

        patients_with_hits = 0
        max_intersection = 0

        for sample_id, genes_samples in sample_gene_map.items():
            if sample_id not in sample_by_id.index:
                continue
            srow = sample_by_id.loc[sample_id]
            patient_id = str(srow[selection.patient_id_col])
            if patient_id not in patient_by_id.index:
                continue
            prow = patient_by_id.loc[patient_id]
            neither = len(universe_genes - genes_samples - genes_pathway)
            anotb = len(genes_samples - genes_pathway)
            bnota = len(genes_pathway - genes_samples)
            both = len(genes_samples & genes_pathway)
            result = significant_intersection(neither, anotb, bnota, both)
            if result["pValue_raw"] <= cfg.fisher_pvalue_threshold and result["OR"] > cfg.fisher_or_threshold and both > 0:
                value = int(both)
                previous_value = patient_vs_pathways[output_name].get(patient_id)
                if previous_value is None or value > previous_value:
                    patient_vs_pathways[output_name][patient_id] = value
                patients_with_hits += 1
                max_intersection = max(max_intersection, value)
                df_out["PatientID"].append(patient_id)
                df_out["SampleID"].append(str(sample_id))
                df_out["PathwayRoot"].append(root_name)
                df_out["PathwayName"].append(pathway_name)
                df_out["PathwayLevel"].append(level)
                sample_values = _series_to_clean_dict(srow, out_sample_cols)
                patient_values = _series_to_clean_dict(prow, out_patient_cols)
                for c in out_sample_cols:
                    df_out[f"SAMPLE__{c}"].append(sample_values[c])
                for c in out_patient_cols:
                    df_out[f"PATIENT__{c}"].append(patient_values[c])
                df_out[output_name].append(value)

        patient_count = len(patient_vs_pathways[output_name])
        if patient_count >= min_patients:
            df_dataset = pd.DataFrame(df_out)
            out_path = datasets_dir / f"{safe_filename(output_name)}.tsv"
            df_dataset.to_csv(out_path, sep="\t", index=False)
            datasets_written.append(out_path)
            datasets[output_name] = df_dataset
            group_resume_records.append(
                {
                    "Pathway": output_name,
                    "PathwayRoot": root_name,
                    "PathwayName": pathway_name,
                    "PathwayLevel": level,
                    "Pathway len": int(len(genes_pathway)),
                    "Patients": int(patient_count),
                    "MaxIntersection": int(max_intersection),
                }
            )

    for pathway_info in pathway_catalog:
        _evaluate_pathway(pathway_info)

    group_resume_df = pd.DataFrame(group_resume_records)
    if group_resume_df.empty:
        group_resume_df = pd.DataFrame(
            columns=["Pathway", "PathwayRoot", "PathwayName", "PathwayLevel", "Pathway len", "Patients", "MaxIntersection"]
        )
    else:
        group_resume_df = group_resume_df.sort_values(
            ["Patients", "Pathway len", "Pathway"], ascending=[False, False, True], ignore_index=True
        )

    group_resume_path = cfg.paths.tables_dir / "group_resume.tsv"
    group_resume_df.to_csv(group_resume_path, sep="\t", index=False)

    combined_df = None
    combined_df_path = None
    if datasets:
        patient_ids = sorted({pid for pathway_map in patient_vs_pathways.values() for pid in pathway_map.keys()})
        combined_base = pd.DataFrame(index=patient_ids)
        combined_base.index.name = selection.patient_id_col

        patient_features = patient.set_index(selection.patient_id_col, drop=True)[out_patient_cols].copy() if out_patient_cols else pd.DataFrame(index=patient_by_id.index)

        sample_for_patient = sample.drop_duplicates(subset=[selection.patient_id_col], keep="first").set_index(selection.patient_id_col, drop=True)
        sample_features = sample_for_patient[out_sample_cols].copy() if out_sample_cols else pd.DataFrame(index=sample_for_patient.index)
        if selection.sample_id_col in sample_for_patient.columns:
            sample_features.insert(0, "SampleID", sample_for_patient[selection.sample_id_col])

        combined_df = combined_base.join(patient_features.add_prefix("PATIENT__"), how="left")
        combined_df = combined_df.join(sample_features.add_prefix("SAMPLE__"), how="left")

        for pathway_name in sorted(datasets.keys()):
            values = pd.Series(patient_vs_pathways[pathway_name], name=pathway_name, dtype="Int64")
            combined_df = combined_df.join(values, how="left")
            combined_df[pathway_name] = combined_df[pathway_name].astype("Int64")

        combined_df_path = datasets_dir / "CombinedDF.tsv"
        combined_df.to_csv(combined_df_path, sep="\t", index=True)

    return {
        "maf_filtered": maf_filtered,
        "sample": sample,
        "patient": patient,
        "group_resume": group_resume_df,
        "group_resume_path": group_resume_path,
        "datasets_dir": datasets_dir,
        "dataset_paths": datasets_written,
        "combined_df": combined_df,
        "combined_df_path": combined_df_path,
        "patient_vs_pathways": patient_vs_pathways,
        "pathway_catalog": pd.DataFrame(pathway_catalog),
    }
