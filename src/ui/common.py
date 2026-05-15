from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from src.domain.models import ParameterSchema, RuntimePaths
from src.services.io_utils import inspect_table, read_table, safe_filename, write_uploaded_bytes

ID_CANDIDATES = {
    "patient": ["PATIENT_ID", "Patient_ID", "patient_id", "PATIENT", "case_id"],
    "sample": ["SAMPLE_ID", "Sample_ID", "sample_id", "Tumor_Sample_Barcode", "SAMPLE"],
    "gene": ["Hugo_Symbol", "Gene", "gene_symbol", "SYMBOL"],
}


def save_upload(uploaded_file, prefix: str, paths: RuntimePaths) -> Path:
    original_name = Path(uploaded_file.name)
    stem = original_name.stem
    suffix = original_name.suffix  # inclui o ponto, e.g. ".tsv" ou ""

    file_id = getattr(uploaded_file, "file_id", "") or ""
    signature = f"{stem}_{uploaded_file.size}_{file_id}"
    base_name = safe_filename(f"{prefix}_{signature}")

    # Re-anexa o suffix DEPOIS de passar pelo safe_filename, para garantir
    # que o ``.`` da extensão não seja transformado por sanitização.
    filename = f"{base_name}{suffix}" if suffix else base_name

    destination = paths.workdir / "uploads" / filename
    return write_uploaded_bytes(uploaded_file.getvalue(), destination)


def inspect_uploaded_table(path_str: str):
    return inspect_table(Path(path_str))


def preview_dataset(path_str: str, max_rows: int = 25) -> pd.DataFrame:
    return read_table(Path(path_str), nrows=max_rows).head(max_rows)


def load_dataset_columns(path_str: str) -> list[str]:
    return list(read_table(Path(path_str), nrows=5).columns)


def guess_column(columns: Iterable[str], role: str) -> str | None:
    cols = list(columns)
    for candidate in ID_CANDIDATES.get(role, []):
        if candidate in cols:
            return candidate
    return cols[0] if cols else None


def default_maf_selection(columns: list[str]) -> list[str]:
    preferred_aliases = [
        ["Hugo_Symbol", "Hugo Symbol"],
        ["Chromosome"],
        ["Start_Position", "Start Position"],
        ["End_Position", "End Position"],
        ["Reference_Allele", "Reference Allele"],
        ["Tumor_Seq_Allele2", "Tumor_Seq_Allele", "Tumor Seq Allele2", "Tumor Seq Allele"],
        ["Variant_Classification", "Variant Classification"],
        ["Variant_Type", "Variant Type"],
        ["Tumor_Sample_Barcode", "Tumor Sample Barcode"],
    ]

    normalized_to_original = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in columns
    }

    selected: list[str] = []

    for aliases in preferred_aliases:
        for alias in aliases:
            key = alias.strip().lower().replace(" ", "_")
            if key in normalized_to_original:
                selected.append(normalized_to_original[key])
                break

    return selected


def render_parameter_input(param: ParameterSchema, scope_key: str):
    widget_key = f"{scope_key}_{param.key}"
    help_text = param.help_text or None
    if param.kind == "bool":
        return st.checkbox(param.label, value=bool(param.default), help=help_text, key=widget_key)
    if param.kind == "int":
        min_value = int(param.min_value if param.min_value is not None else 0)
        max_value = int(param.max_value if param.max_value is not None else 100000)
        step = int(param.step if param.step is not None else 1)
        return st.number_input(param.label, value=int(param.default), min_value=min_value, max_value=max_value, step=step, help=help_text, key=widget_key)
    if param.kind == "float":
        min_value = float(param.min_value if param.min_value is not None else 0.0)
        max_value = float(param.max_value if param.max_value is not None else 1e9)
        step = float(param.step if param.step is not None else 0.1)
        return st.number_input(param.label, value=float(param.default), min_value=min_value, max_value=max_value, step=step, help=help_text, key=widget_key)
    if param.kind == "select":
        options = param.options or [param.default]
        try:
            default_index = options.index(param.default)
        except ValueError:
            default_index = 0
        return st.selectbox(param.label, options=options, index=default_index, help=help_text, key=widget_key)
    return st.text_input(param.label, value=str(param.default), help=help_text, key=widget_key)


def render_parameter_schema_list(parameters: list[ParameterSchema], scope_key: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for param in parameters:
        values[param.key] = render_parameter_input(param, scope_key)
    return values


def parse_json_text(text: str, label: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em {label}: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} precisa ser um objeto JSON.")
    return parsed


def render_section_header(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def render_info_card(title: str, lines: list[str]) -> None:
    body = "".join([f"<div>{line}</div>" for line in lines])
    st.markdown(f'<div class="causal-card"><strong>{title}</strong>{body}</div>', unsafe_allow_html=True)


def guess_outcome_column(columns: list[str]) -> str | None:
    preferred = [
        "OS_STATUS",
        "OS Status",
        "os_status",
        "OS_STATUS_BINARY",
        "OS Status Binary",
        "SURVIVAL_STATUS",
        "Survival Status",
        "survival_status",
        "VITAL_STATUS",
        "Vital Status",
        "vital_status",
        "STATUS",
        "Status",
        "status",
    ]

    normalized_to_original = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in columns
    }

    for name in preferred:
        key = name.strip().lower().replace(" ", "_")
        if key in normalized_to_original:
            return normalized_to_original[key]

    for col in columns:
        normalized = str(col).strip().lower()
        if "os_status" in normalized:
            return col
        if "survival" in normalized and "status" in normalized:
            return col
        if "vital" in normalized and "status" in normalized:
            return col
        if normalized == "status":
            return col

    return None