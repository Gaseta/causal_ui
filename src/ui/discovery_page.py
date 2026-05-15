from __future__ import annotations

from pathlib import Path
from time import monotonic

import streamlit as st

from src.domain.models import DiscoveryRunConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.discovery_service import run_discovery
from src.services.parameter_registry import DISCOVERY_ALGORITHMS
from src.ui.common import (
    load_dataset_columns,
    preview_dataset,
    render_parameter_schema_list,
    render_section_header,
    save_upload,
)


# Limites para evitar estouro de memória se o log crescer demais.
_MAX_HISTORY_MESSAGES = 100
_RENDER_TAIL = 50


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes:02d}min {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}min {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _render_progress_panel(container, messages: list, current_message: str) -> None:
    """Renderiza o histórico recente do progresso num único bloco de texto.

    Mesma estratégia usada no preprocessing_page: economia de memória ao
    evitar criar um widget Streamlit por mensagem.
    """
    container.empty()
    tail = messages[-_RENDER_TAIL:]
    if not tail:
        return
    with container.container():
        st.markdown("**Histórico recente:**")
        st.code("\n".join(tail), language="text")


def render(paths: RuntimePaths) -> None:
    st.subheader("Descoberta causal")
    render_section_header(
        "1. Dataset",
        "Você pode usar o CombinedDF gerado no pré-processamento ou anexar um dataset já pronto.",
    )

    dataset_upload = st.file_uploader(
        "Dataset para descoberta causal (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
        key="ui_discovery_dataset",
    )

    dataset_path: Path | None = None
    if dataset_upload:
        dataset_path = Path(save_upload(dataset_upload, "discovery_dataset", paths))
    elif (
        st.session_state.get("preprocess_result")
        and st.session_state["preprocess_result"].get("combined_df_path")
    ):
        dataset_path = Path(st.session_state["preprocess_result"]["combined_df_path"])
        st.info(f"Usando CombinedDF do passo 1: {dataset_path}")

    if dataset_path is not None:
        st.dataframe(
            preview_dataset(str(dataset_path), max_rows=15),
            use_container_width=True,
            height=280,
        )
        st.caption(f"Colunas detectadas: {len(load_dataset_columns(str(dataset_path)))}")

    grouped: dict = {}
    for spec in DISCOVERY_ALGORITHMS:
        grouped.setdefault(spec.family, []).append(spec)

    render_section_header("2. Algoritmo e parâmetros")
    family = st.selectbox(
        "Família de algoritmos",
        options=list(grouped.keys()),
        key="ui_discovery_family",
    )
    spec = st.selectbox(
        "Algoritmo",
        options=grouped[family],
        format_func=lambda x: x.label,
        key="ui_discovery_algorithm",
    )
    st.caption(spec.description)
    params = render_parameter_schema_list(spec.parameters, f"discovery_{spec.key}")

    # Aviso específico por tipo de backend declarado no registry.
    if spec.backend == "placeholder":
        st.warning(
            f"O algoritmo **{spec.label}** está catalogado na UI, mas precisa de uma "
            "ponte específica do py-tetrad implementada no ambiente alvo. "
            "Execução vai resultar em erro até que essa ponte exista — use PC, FGES, "
            "GES ou FCI por enquanto."
        )
    elif spec.backend == "pytetrad":
        st.info(
            f"O algoritmo **{spec.label}** requer py-tetrad instalado e configurado. "
            "Não há fallback para causal-learn neste algoritmo: se py-tetrad não estiver "
            "disponível, a execução irá falhar."
        )

    st.markdown("### Andamento da descoberta causal")
    progress_bar_area = st.empty()
    progress_log_area = st.empty()

    run_disabled = dataset_path is None
    run_clicked = st.button(
        "Executar descoberta causal",
        type="primary",
        key="run_discovery",
        disabled=run_disabled,
    )

    if run_clicked and dataset_path is not None:
        progress_widget = progress_bar_area.progress(
            0,
            text="Iniciando descoberta causal...",
        )
        progress_messages: list = []
        started_at = monotonic()

        def _ui_progress(message: str, progress: float | None = None) -> None:
            elapsed = monotonic() - started_at
            if progress is not None:
                pct = int(max(0.0, min(1.0, float(progress))) * 100)
                progress_widget.progress(
                    pct,
                    text=(
                        "Descoberta causal em andamento... "
                        f"tempo decorrido: {_format_elapsed(elapsed)}"
                    ),
                )

            if not message:
                return

            progress_messages.append(message)
            if len(progress_messages) > _MAX_HISTORY_MESSAGES:
                del progress_messages[: len(progress_messages) - _MAX_HISTORY_MESSAGES]
            _render_progress_panel(progress_log_area, progress_messages, message)

        try:
            result = run_discovery(
                DiscoveryRunConfig(
                    dataset_path=dataset_path,
                    algorithm_key=spec.key,
                    params=params,
                    paths=paths,
                ),
                progress_callback=_ui_progress,
            )
            total_seconds = monotonic() - started_at
            progress_widget.progress(
                100,
                text=f"Descoberta concluída. Tempo total: {_format_elapsed(total_seconds)}",
            )
            st.session_state.discovery_result = result
            st.success(
                f"Descoberta causal concluída com {len(result.get('edges', []))} aresta(s) "
                f"(engine: **{result.get('engine_used', 'unknown')}**)."
            )
        except RuntimeError as exc:
            # Erros previsíveis: placeholder, py-tetrad indisponível para algoritmo
            # sem fallback, etc.
            st.error(f"Não foi possível executar a descoberta causal: {exc}")
        except Exception as exc:
            # Erros inesperados — mostra mensagem amigável + permite ver
            # detalhes técnicos sob demanda.
            st.error(f"Falha na descoberta causal: {type(exc).__name__}: {exc}")
            with st.expander("Detalhes técnicos do erro"):
                import traceback as _tb
                st.code(_tb.format_exc(), language="text")

    result = st.session_state.get("discovery_result")
    if not result:
        return

    render_section_header("3. Resultados")

    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Variáveis usadas", len(result.get("columns", [])))
    with meta_cols[1]:
        st.metric("Arestas", len(result.get("edges", [])))
    with meta_cols[2]:
        directed_count = len(result.get("directed_edges", []))
        st.metric("Arestas direcionadas", directed_count)
    with meta_cols[3]:
        st.metric("Engine", result.get("engine_used", "-"))

    st.caption(f"Algoritmo: **{Path(result['output_dir']).name}**")

    output_paths_cols = st.columns(2)
    with output_paths_cols[0]:
        st.markdown("**Arquivo de arestas (TETRAD):**")
        st.code(str(result["edges_path"]), language="text")
    with output_paths_cols[1]:
        st.markdown("**Arquivo DOT (GraphViz):**")
        st.code(str(result["dot_path"]), language="text")

    if result.get("image_path") and Path(result["image_path"]).exists():
        st.image(str(result["image_path"]), caption="Rede causal plotada")
    elif result.get("edges"):
        st.info(
            "Imagem do grafo não foi renderizada (GraphViz indisponível ou arestas "
            "insuficientes). O arquivo DOT está disponível para visualização externa."
        )

    with st.expander("Arestas em formato TETRAD", expanded=False):
        edges = result.get("edges", [])
        st.text("\n".join(edges) if edges else "Nenhuma aresta retornada.")

    with st.expander("Diagnóstico da execução (debug)", expanded=False):
        run_debug_path = result.get("run_debug_path")
        if run_debug_path and Path(run_debug_path).exists():
            st.code(Path(run_debug_path).read_text(encoding="utf-8"), language="json")

    discovery_zip = zip_paths(
        paths.workdir / "causal_discovery_outputs.zip",
        [result["output_dir"]],
        base_dir=paths.workdir,
    )
    st.download_button(
        "Baixar ZIP da descoberta causal",
        data=discovery_zip.read_bytes(),
        file_name=discovery_zip.name,
        mime="application/zip",
        key="download_discovery_zip",
    )