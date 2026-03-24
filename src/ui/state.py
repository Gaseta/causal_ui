from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.domain.models import RuntimePaths
from src.services.io_utils import temp_workdir


SESSION_DEFAULTS = {
    "preprocess_result": None,
    "discovery_result": None,
    "inference_result": None,
    "ui_messages": [],
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def build_runtime_paths() -> RuntimePaths:
    project_root = Path(__file__).resolve().parents[2]
    workdir = temp_workdir()
    return RuntimePaths.build(project_root=project_root, workdir=workdir)
