from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.domain.models import InferenceRunConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.inference_service import run_inference
from src.services.parameter_registry import ESTIMATE_METHODS, ESTIMATE_METHOD_SPECS, REFUTERS, REFUTER_SPECS
from src.ui.common import load_dataset_columns, parse_json_text, preview_dataset, render_parameter_schema_list, render_section_header, save_upload



def _default_pair_choices(columns: list[str]) -> tuple[list[str], list[str]]:
    if not columns:
        return [], []
    if len(columns) == 1:
        return [columns[0]], []
    return [columns[0]], [columns[1]]



def render(paths: RuntimePaths) -> None:
    st.subheader("Inferência causal e refutação")
    render_section_header(
        "1. Entradas",
        "Use o dataset e o grafo gerados nos passos anteriores ou envie seus próprios arquivos para executar apenas a inferência.",
    )

    dataset_upload = st.file_uploader("Dataset para inferência (CSV/TSV/TXT)", type=["csv", "tsv", "txt"], key="ui_inference_dataset")
    graph_upload = st.file_uploader("Modelo causal (DOT ou edges TETRAD)", type=["dot", "txt"], key="ui_inference_graph")

    dataset_path = None
    graph_path = None
    if dataset_upload:
        dataset_path = save_upload(dataset_upload, "inference_dataset", paths)
    elif st.session_state.get("preprocess_result") and st.session_state["preprocess_result"].get("combined_df_path"):
        dataset_path = st.session_state["preprocess_result"]["combined_df_path"]
        st.info(f"Usando CombinedDF do passo 1: {dataset_path}")

    if graph_upload:
        graph_path = save_upload(graph_upload, "inference_graph", paths)
    elif st.session_state.get("discovery_result") and st.session_state["discovery_result"].get("dot_path"):
        graph_path = st.session_state["discovery_result"]["dot_path"]
        st.info(f"Usando grafo do passo 2: {graph_path}")

    columns: list[str] = []
    if dataset_path:
        st.dataframe(preview_dataset(str(dataset_path), max_rows=15), use_container_width=True, height=260)
        columns = load_dataset_columns(str(dataset_path))

    render_section_header("2. Método de estimativa")
    estimate_family = st.selectbox("Família de métodos DoWhy", options=list(ESTIMATE_METHODS.keys()), key="ui_estimate_family")
    estimate_method = st.selectbox("Método de estimativa", options=ESTIMATE_METHODS[estimate_family], key="ui_estimate_method")
    estimate_params = render_parameter_schema_list(ESTIMATE_METHOD_SPECS.get(estimate_method, []), f"estimate_{estimate_method}")
    estimate_extra_text = st.text_area("Parâmetros extras do método em JSON", value="{}", key="ui_estimate_params_json")

    render_section_header("3. Treatment e outcome")
    all_pairs = st.checkbox("Testar todas as combinações possíveis de treatment e outcome", value=False, key="ui_all_pairs")
    default_treatments, default_outcomes = _default_pair_choices(columns)
    treatment_cols = st.multiselect("Treatment(s)", options=columns, default=default_treatments, key="ui_treatments")
    outcome_cols = st.multiselect("Outcome(s)", options=columns, default=default_outcomes, key="ui_outcomes")

    render_section_header("4. Refutação")
    refuters = st.multiselect("Métodos de refutação", options=REFUTERS, default=REFUTERS[:3], key="ui_refuters")
    refuter_params: dict[str, dict] = {}
    for refuter in refuters:
        with st.expander(f"Parâmetros: {refuter}", expanded=False):
            refuter_params[refuter] = render_parameter_schema_list(REFUTER_SPECS.get(refuter, []), f"refuter_{refuter}")
    refuter_extra_text = st.text_area("Parâmetros extras dos refuters em JSON por nome", value="{}", key="ui_refuters_json")

    if dataset_path and graph_path and st.button("Executar inferência causal", type="primary", key="run_inference"):
        try:
            estimate_extra = parse_json_text(estimate_extra_text, "parâmetros extras do método")
            refuter_extra = parse_json_text(refuter_extra_text, "parâmetros extras dos refuters")
            merged_estimate_params = {**estimate_params, **estimate_extra}
            merged_refuter_params = {**refuter_params}
            for ref_name, ref_cfg in refuter_extra.items():
                current = merged_refuter_params.get(ref_name, {})
                if isinstance(ref_cfg, dict):
                    merged_refuter_params[ref_name] = {**current, **ref_cfg}
        except ValueError as e:
            st.error(str(e))
            return

        with st.spinner("Executando inferência causal..."):
            result = run_inference(
                InferenceRunConfig(
                    dataset_path=Path(dataset_path),
                    graph_path=Path(graph_path),
                    estimate_method=estimate_method,
                    estimate_params=merged_estimate_params,
                    refuters=refuters,
                    refuter_params=merged_refuter_params,
                    treatment_columns=treatment_cols,
                    outcome_columns=outcome_cols,
                    all_pairs=all_pairs,
                    paths=paths,
                )
            )
        st.session_state.inference_result = result
        st.success("Inferência concluída.")

    result = st.session_state.get("inference_result")
    if not result:
        return

    render_section_header("5. Resultados")
    st.metric("Pares avaliados", len(result.get("pairs", [])))
    st.json(result)

    result_dir = paths.causal_results_dir / "inference"
    inference_zip = zip_paths(paths.workdir / "causal_inference_outputs.zip", [result_dir], base_dir=paths.workdir)
    st.download_button(
        "Baixar ZIP da inferência",
        data=inference_zip.read_bytes(),
        file_name=inference_zip.name,
        mime="application/zip",
        key="download_inference_zip",
    )
