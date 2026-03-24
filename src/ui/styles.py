from __future__ import annotations

import streamlit as st


APP_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.2rem;}
.causal-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    padding: 1rem 1rem .8rem 1rem;
    background: rgba(250,250,250,.45);
    margin-bottom: .9rem;
}
.causal-muted {opacity: .8; font-size: .92rem;}
.small-code {font-size: .85rem;}
</style>
"""


def apply_global_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
