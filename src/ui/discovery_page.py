from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.domain.models import DiscoveryRunConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.discovery_service import run_discovery
from src.services.parameter_registry import DISCOVERY_ALGORITHMS
from src.ui.common import load_dataset_columns, preview_dataset, render_parameter_schema_list, render_section_header, save_upload



def render(paths: RuntimePaths) -> None:
    st.subheader("Descoberta causal")
    render_section_header(
        "1. Dataset",
        "Você pode usar o CombinedDF gerado no pré-processamento ou anexar um dataset já pronto.",
    )

    dataset_upload = st.file_uploader("Dataset para descoberta causal (CSV/TSV/TXT)", type=["csv", "tsv", "txt"], key="ui_discovery_dataset")
    dataset_path = None
    if dataset_upload:
        dataset_path = save_upload(dataset_upload, "discovery_dataset", paths)
    elif st.session_state.get("preprocess_result") and st.session_state["preprocess_result"].get("combined_df_path"):
        dataset_path = st.session_state["preprocess_result"]["combined_df_path"]
        st.info(f"Usando CombinedDF do passo 1: {dataset_path}")

    if dataset_path:
        st.dataframe(preview_dataset(str(dataset_path), max_rows=15), use_container_width=True, height=280)
        st.caption(f"Colunas detectadas: {len(load_dataset_columns(str(dataset_path)))}")

    grouped: dict[str, list] = {}
    for spec in DISCOVERY_ALGORITHMS:
        grouped.setdefault(spec.family, []).append(spec)

    render_section_header("2. Algoritmo e parâmetros")
    family = st.selectbox("Família de algoritmos", options=list(grouped.keys()), key="ui_discovery_family")
    spec = st.selectbox("Algoritmo", options=grouped[family], format_func=lambda x: x.label, key="ui_discovery_algorithm")
    st.caption(spec.description)
    params = render_parameter_schema_list(spec.parameters, f"discovery_{spec.key}")

    if spec.backend == "placeholder":
        st.warning("Este algoritmo está catalogado na UI, mas a ponte py-tetrad ainda precisa de implementação específica no ambiente alvo.")

    if dataset_path and st.button("Executar descoberta causal", type="primary", key="run_discovery"):
        with st.spinner("Executando descoberta causal..."):
            result = run_discovery(
                DiscoveryRunConfig(
                    dataset_path=Path(dataset_path),
                    algorithm_key=spec.key,
                    params=params,
                    paths=paths,
                )
            )
        st.session_state.discovery_result = result
        st.success("Descoberta causal concluída.")

    result = st.session_state.get("discovery_result")
    if not result:
        return

    render_section_header("3. Resultados")
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Variáveis usadas", len(result.get("columns", [])))
    with meta_cols[1]:
        st.metric("Edges", len(result.get("edges", [])))
    with meta_cols[2]:
        st.metric("Algoritmo", Path(result.get("output_dir")).name if result.get("output_dir") else "-")

    st.code(str(result["edges_path"]), language="text")
    st.code(str(result["dot_path"]), language="text")
    if result.get("image_path") and Path(result["image_path"]).exists():
        st.image(str(result["image_path"]), caption="Rede causal plotada")

    with st.expander("Edges em formato TETRAD", expanded=False):
        st.text("\n".join(result.get("edges", [])) or "Nenhuma aresta retornada.")

    discovery_zip = zip_paths(paths.workdir / "causal_discovery_outputs.zip", [result["output_dir"]], base_dir=paths.workdir)
    st.download_button(
        "Baixar ZIP da descoberta causal",
        data=discovery_zip.read_bytes(),
        file_name=discovery_zip.name,
        mime="application/zip",
        key="download_discovery_zip",
    )
