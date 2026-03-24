from __future__ import annotations

import streamlit as st

from src.ui import discovery_page, inference_page, preprocessing_page
from src.ui.state import build_runtime_paths, init_session_state
from src.ui.styles import apply_global_styles



def render_app() -> None:
    st.set_page_config(page_title="Causal Discovery & Inference UI", layout="wide")
    apply_global_styles()
    init_session_state()
    paths = build_runtime_paths()

    st.title("Causal Discovery & Inference UI")
    st.caption("Aplicação em Streamlit para pré-processamento, descoberta causal e inferência causal, organizada em etapas e conectada ao backend Python.")

    with st.sidebar:
        st.markdown("## Navegação")
        st.write("Use as abas centrais para executar o pipeline completo ou apenas uma etapa.")
        st.code(str(paths.workdir), language="text")

    tab1, tab2, tab3 = st.tabs([
        "1. Pré-processamento",
        "2. Descoberta causal",
        "3. Inferência causal",
    ])

    with tab1:
        preprocessing_page.render(paths)
    with tab2:
        discovery_page.render(paths)
    with tab3:
        inference_page.render(paths)
