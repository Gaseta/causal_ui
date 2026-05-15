from __future__ import annotations

from pathlib import Path
from time import monotonic

import pandas as pd
import streamlit as st

from src.domain.models import InferenceRunConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.inference_service import run_inference
from src.services.parameter_registry import (
    ESTIMATE_METHODS,
    ESTIMATE_METHOD_SPECS,
    REFUTERS,
    REFUTER_SPECS,
)
from src.ui.common import (
    guess_outcome_column,
    load_dataset_columns,
    parse_json_text,
    preview_dataset,
    render_parameter_schema_list,
    render_section_header,
    save_upload,
)


# Limites para evitar estouro de memória se o log crescer demais.
_MAX_HISTORY_MESSAGES = 100
_RENDER_TAIL = 50


# Texto explicativo dos refuters — usado em tooltips e seções de ajuda.
_REFUTER_EXPLANATIONS = {
    "placebo_treatment_refuter": {
        "what": "Substitui o tratamento por uma variável aleatória (placebo).",
        "expectation": "O efeito estimado deveria ir a zero.",
        "interpretation": (
            "p ≥ 0.05 → estimador passou (não conseguimos rejeitar a hipótese "
            "de que o efeito é zero). "
            "p < 0.05 → estimador FALHOU (encontrou um efeito espúrio significativo "
            "quando não deveria existir)."
        ),
    },
    "random_common_cause": {
        "what": "Adiciona uma variável aleatória independente como causa comum.",
        "expectation": "O efeito não deveria mudar (a variável aleatória não afeta nada).",
        "interpretation": (
            "p ≥ 0.05 → estimador é robusto (passou). "
            "p < 0.05 → estimador é sensível a variáveis irrelevantes, possivelmente "
            "indicando viés."
        ),
    },
    "data_subset_refuter": {
        "what": "Re-estima o efeito em uma subamostra dos dados.",
        "expectation": "O efeito deveria ser similar ao original.",
        "interpretation": (
            "p ≥ 0.05 → estimador é robusto à variação amostral (passou). "
            "p < 0.05 → o efeito muda drasticamente com subsamples, sugerindo "
            "instabilidade do estimador."
        ),
    },
    "add_unobserved_common_cause": {
        "what": "Adiciona um confundidor não-observado simulado entre tratamento e outcome.",
        "expectation": "Análise de sensibilidade — não há p-valor com veredicto binário.",
        "interpretation": (
            "Quanto mais próximo o novo efeito estiver do original, mais robusta é a "
            "estimativa à possibilidade de confundidores ocultos. Use o intervalo de "
            "new_effect para entender a sensibilidade."
        ),
    },
}


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
    container.empty()
    tail = messages[-_RENDER_TAIL:]
    if not tail:
        return
    with container.container():
        st.markdown("**Histórico recente:**")
        st.code("\n".join(tail), language="text")


def _guess_default_treatment(columns: list) -> list:
    """Sugere uma coluna razoável para treatment.

    Preferimos colunas clínicas modificáveis ou clínicas associadas a
    intervenção (TMB, idade, estágio). Evitamos OS_STATUS (que costuma ser
    outcome).
    """
    preferred = [
        "TMB_NONSYNONYMOUS",
        "TUMOR_STAGE",
        "AGE_AT_DIAGNOSIS",
        "INFERRED_MENOPAUSAL_STATE",
    ]
    for p in preferred:
        if p in columns:
            return [p]
    outcome_guess = guess_outcome_column(columns)
    for c in columns:
        if c != outcome_guess:
            return [c]
    return [columns[0]] if columns else []


def _guess_default_outcome(columns: list) -> list:
    """Sugere uma coluna razoável para outcome (geralmente OS_STATUS)."""
    guess = guess_outcome_column(columns)
    if guess:
        return [guess]
    return [columns[1]] if len(columns) >= 2 else []


# Nomes (case-insensitive) que identificam colunas de ID/identificador no
# CombinedDF. Essas colunas NUNCA devem ser tratadas como variáveis causais
# (treatment ou outcome), porque são apenas identificadores únicos.
_ID_COLUMN_NAMES = {
    "patient_id", "sample_id", "tumor_sample_barcode", "case_id", "id",
}


def _filter_id_columns(columns: list) -> list:
    """Remove colunas de ID (PATIENT_ID, SAMPLE_ID, etc.) da lista.

    Usado quando o usuário marca "Testar todas as combinações" e queremos
    expandir automaticamente para todas as colunas — sem incluir IDs, que
    produziriam apenas pares pulados.
    """
    return [c for c in columns if str(c).strip().lower() not in _ID_COLUMN_NAMES]


def _render_estimate_card(pair: dict) -> None:
    """Renderiza um par individual de forma rica e interpretável."""
    treatment = pair.get("treatment", "")
    outcome = pair.get("outcome", "")
    estimate_value = pair.get("estimate_value")
    backdoor_vars = pair.get("backdoor_vars") or []
    realized_expr = pair.get("realized_expression")
    target_units = pair.get("target_units") or "ate"
    refuters = pair.get("refuters") or []

    st.markdown(f"#### {treatment}  →  {outcome}")
    col_main, col_meta = st.columns([1, 1])
    with col_main:
        if estimate_value is not None:
            st.metric(
                label=f"Effect estimate ({target_units.upper()})",
                value=f"{estimate_value:.4f}",
                help=(
                    "Valor estimado do efeito causal médio. "
                    f"Métrica: {target_units.upper()} "
                    "(ate=average treatment effect, "
                    "att=avg treatment on treated, "
                    "atc=avg treatment on control)."
                ),
            )
        else:
            st.metric(label="Effect estimate", value="—")

        if realized_expr:
            st.caption(f"Expressão estimada: `{realized_expr}`")

    with col_meta:
        if backdoor_vars:
            st.markdown("**Variáveis de ajuste (backdoor):**")
            st.write(", ".join(backdoor_vars))
        else:
            st.caption("Nenhuma variável de backdoor identificada.")

        ci = pair.get("confidence_intervals")
        if ci:
            st.caption(f"Intervalo de confiança: `{ci}`")

    if refuters:
        st.markdown("**Testes de robustez (refuters):**")
        rows = []
        for r in refuters:
            passes = r.get("passes")
            if r.get("error"):
                status = "Erro na execução"
            elif passes is True:
                status = "Passou"
            elif passes is False:
                status = "Falhou"
            else:
                status = "Sem critério"

            new_effect = r.get("new_effect")
            if isinstance(new_effect, list):
                new_effect_str = (
                    f"[{new_effect[0]:.4f}, {new_effect[1]:.4f}]"
                    if len(new_effect) == 2
                    and all(isinstance(v, (int, float)) for v in new_effect)
                    else str(new_effect)
                )
            elif isinstance(new_effect, (int, float)):
                new_effect_str = f"{new_effect:.4f}"
            else:
                new_effect_str = "—"

            p_value = r.get("p_value")
            p_value_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "—"

            rows.append({
                "Refuter": r.get("refuter_name", ""),
                "Status": status,
                "Novo efeito": new_effect_str,
                "p-value": p_value_str,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Como interpretar cada refuter?", expanded=False):
            for r in refuters:
                name = r.get("refuter_name", "")
                explanation = _REFUTER_EXPLANATIONS.get(name)
                if not explanation:
                    continue
                st.markdown(f"**{name}**")
                st.markdown(f"- O que faz: {explanation['what']}")
                st.markdown(f"- Comportamento esperado: {explanation['expectation']}")
                st.markdown(f"- Interpretação: {explanation['interpretation']}")
                if r.get("error"):
                    st.markdown(f"- *Erro:* `{r['error']}`")
                st.markdown("---")

    st.caption(f"Arquivos detalhados: `{pair.get('pair_dir', '')}`")


def render(paths: RuntimePaths) -> None:
    st.subheader("Inferência causal e refutação")
    render_section_header(
        "1. Entradas",
        "Use o dataset e o grafo gerados nos passos anteriores ou envie seus próprios arquivos para executar apenas a inferência.",
    )

    dataset_upload = st.file_uploader(
        "Dataset para inferência (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
        key="ui_inference_dataset",
    )
    graph_upload = st.file_uploader(
        "Modelo causal (DOT ou edges TETRAD)",
        type=["dot", "txt"],
        key="ui_inference_graph",
    )

    dataset_path: Path | None = None
    graph_path: Path | None = None

    if dataset_upload:
        dataset_path = Path(save_upload(dataset_upload, "inference_dataset", paths))
    elif (
        st.session_state.get("preprocess_result")
        and st.session_state["preprocess_result"].get("combined_df_path")
    ):
        dataset_path = Path(st.session_state["preprocess_result"]["combined_df_path"])
        st.info(f"Usando CombinedDF do passo 1: {dataset_path}")

    if graph_upload:
        graph_path = Path(save_upload(graph_upload, "inference_graph", paths))
    elif (
        st.session_state.get("discovery_result")
        and st.session_state["discovery_result"].get("dot_path")
    ):
        graph_path = Path(st.session_state["discovery_result"]["dot_path"])
        st.info(f"Usando grafo do passo 2: {graph_path}")

    columns: list = []
    if dataset_path is not None:
        st.dataframe(
            preview_dataset(str(dataset_path), max_rows=15),
            use_container_width=True,
            height=260,
        )
        columns = load_dataset_columns(str(dataset_path))

    # ------------------------------------------------------------------
    # 2. Método de estimativa
    # ------------------------------------------------------------------
    render_section_header("2. Método de estimativa")
    estimate_family = st.selectbox(
        "Família de métodos DoWhy",
        options=list(ESTIMATE_METHODS.keys()),
        key="ui_estimate_family",
    )
    estimate_method = st.selectbox(
        "Método de estimativa",
        options=ESTIMATE_METHODS[estimate_family],
        key="ui_estimate_method",
    )
    estimate_params = render_parameter_schema_list(
        ESTIMATE_METHOD_SPECS.get(estimate_method, []),
        f"estimate_{estimate_method}",
    )
    estimate_extra_text = st.text_area(
        "Parâmetros extras do método em JSON",
        value="{}",
        key="ui_estimate_params_json",
        help=(
            "Use para passar parâmetros não cobertos pelos campos acima. "
            "Exemplo: {\"control_value\": 0, \"treatment_value\": 1}"
        ),
    )

    # ------------------------------------------------------------------
    # 3. Treatment e outcome
    # ------------------------------------------------------------------
    render_section_header("3. Treatment e outcome")

    all_pairs = st.checkbox(
        "Testar todas as combinações possíveis de treatment e outcome",
        value=False,
        key="ui_all_pairs",
        help=(
            "Marcado: avalia o produto cartesiano (cada treatment × cada outcome, "
            "exceto pares idênticos). "
            "Desmarcado: avalia pareamento posicional (treatments[i] com outcomes[i]), "
            "truncando no menor dos dois."
        ),
    )

    default_treatments = _guess_default_treatment(columns)
    default_outcomes = _guess_default_outcome(columns)

    treatment_cols = st.multiselect(
        "Treatment(s)" + ("  (opcional)" if all_pairs else ""),
        options=columns,
        default=default_treatments,
        key="ui_treatments",
        help=(
            "Variável(eis) cujo efeito causal queremos estimar. Em medicina, "
            "geralmente é uma variável modificável (tratamento, exposição, dose). "
            "Deve estar presente no grafo causal. "
            "Quando 'Testar todas as combinações' está marcado, deixar em branco "
            "expande para TODAS as colunas do dataset (IDs filtrados)."
        ),
    )
    outcome_cols = st.multiselect(
        "Outcome(s)" + ("  (opcional)" if all_pairs else ""),
        options=columns,
        default=default_outcomes,
        key="ui_outcomes",
        help=(
            "Variável(eis) onde queremos medir o efeito (desfecho). "
            "Deve estar presente no grafo causal. "
            "Quando 'Testar todas as combinações' está marcado, deixar em branco "
            "expande para TODAS as colunas do dataset (IDs filtrados)."
        ),
    )

    # ----- Resolver seleções efetivas. -----
    # Quando ``all_pairs`` está marcado, seleções vazias são expandidas para
    # TODAS as colunas do dataset (com IDs filtrados). Isso permite ao usuário
    # rodar inferência exploratória sem ter que selecionar cada variável.
    if all_pairs:
        all_non_id_columns = _filter_id_columns(columns)
        effective_treatments = treatment_cols if treatment_cols else all_non_id_columns
        effective_outcomes = outcome_cols if outcome_cols else all_non_id_columns
        ids_filtered = [c for c in columns if c not in all_non_id_columns]
        if (not treatment_cols or not outcome_cols) and ids_filtered:
            st.info(
                f"Modo 'todas as combinações' ativo. Como o multiselect está vazio, "
                f"vamos usar **todas as {len(all_non_id_columns)} colunas** do dataset "
                f"(filtradas: {', '.join(ids_filtered)})."
            )
    else:
        effective_treatments = list(treatment_cols)
        effective_outcomes = list(outcome_cols)

    # ----- Validações visíveis das seleções. -----
    selection_errors: list = []
    selection_warnings: list = []

    if not effective_treatments:
        selection_errors.append(
            "Nenhum **treatment** disponível. Adicione ao menos uma variável "
            "ou marque 'Testar todas as combinações' (com dataset não-vazio)."
        )
    if not effective_outcomes:
        selection_errors.append(
            "Nenhum **outcome** disponível. Adicione ao menos uma variável "
            "ou marque 'Testar todas as combinações' (com dataset não-vazio)."
        )

    if effective_treatments and effective_outcomes:
        overlap = set(effective_treatments) & set(effective_outcomes)
        if (
            overlap
            and len(effective_treatments) == 1
            and len(effective_outcomes) == 1
        ):
            selection_errors.append(
                f"Treatment e outcome são a mesma variável (`{list(overlap)[0]}`). "
                "Selecione variáveis diferentes."
            )
        elif overlap and not all_pairs:
            # No modo all_pairs essa sobreposição é esperada (e os pares
            # iguais são filtrados pelo iterator). Avisar só no modo posicional.
            selection_warnings.append(
                f"Atenção: as variáveis `{', '.join(sorted(overlap))}` aparecem em "
                "AMBOS treatment e outcome. Pares onde treatment == outcome "
                "serão automaticamente ignorados."
            )

        if (
            not all_pairs
            and len(effective_treatments) != len(effective_outcomes)
        ):
            selection_warnings.append(
                f"Listas de treatment ({len(effective_treatments)}) e outcome "
                f"({len(effective_outcomes)}) têm tamanhos diferentes — o pareamento "
                f"posicional vai usar apenas os primeiros "
                f"{min(len(effective_treatments), len(effective_outcomes))} pares. "
                "Marque 'Testar todas as combinações' se preferir o produto cartesiano."
            )

    # Estimar número de pares que serão gerados.
    if effective_treatments and effective_outcomes:
        if all_pairs:
            estimated_pairs = sum(
                1 for t in effective_treatments for o in effective_outcomes if t != o
            )
        else:
            estimated_pairs = sum(
                1 for t, o in zip(effective_treatments, effective_outcomes) if t != o
            )
        st.caption(
            f"Serão avaliados **{estimated_pairs}** par(es) "
            f"({'produto cartesiano' if all_pairs else 'pareamento posicional'})."
        )

    for err in selection_errors:
        st.error(err)
    for warn in selection_warnings:
        st.warning(warn)

    # ------------------------------------------------------------------
    # 4. Refutação
    # ------------------------------------------------------------------
    render_section_header("4. Refutação")
    st.caption(
        "Refuters são testes de robustez que verificam se a estimativa causal "
        "é estável a perturbações dos dados. Veja o detalhamento abaixo de "
        "cada um para entender o que significam os resultados."
    )

    refuters = st.multiselect(
        "Métodos de refutação",
        options=REFUTERS,
        default=REFUTERS[:3],
        key="ui_refuters",
    )
    refuter_params: dict = {}
    for refuter in refuters:
        with st.expander(f"Parâmetros: {refuter}", expanded=False):
            explanation = _REFUTER_EXPLANATIONS.get(refuter)
            if explanation:
                st.caption(
                    f"**O que faz:** {explanation['what']} "
                    f"**Esperado:** {explanation['expectation']}"
                )
            refuter_params[refuter] = render_parameter_schema_list(
                REFUTER_SPECS.get(refuter, []), f"refuter_{refuter}",
            )

    refuter_extra_text = st.text_area(
        "Parâmetros extras dos refuters em JSON por nome",
        value="{}",
        key="ui_refuters_json",
        help=(
            "Exemplo: {\"placebo_treatment_refuter\": {\"random_seed\": 42}}"
        ),
    )

    # ------------------------------------------------------------------
    # 5. Execução
    # ------------------------------------------------------------------
    st.markdown("### Andamento da inferência causal")
    progress_bar_area = st.empty()
    progress_log_area = st.empty()

    run_disabled = (
        dataset_path is None
        or graph_path is None
        or bool(selection_errors)
    )
    if dataset_path is None or graph_path is None:
        st.caption("Envie o dataset e o grafo (ou rode os passos anteriores) para habilitar a execução.")

    run_clicked = st.button(
        "Executar inferência causal",
        type="primary",
        key="run_inference",
        disabled=run_disabled,
    )

    if run_clicked and dataset_path is not None and graph_path is not None:
        try:
            estimate_extra = parse_json_text(
                estimate_extra_text, "parâmetros extras do método",
            )
            refuter_extra = parse_json_text(
                refuter_extra_text, "parâmetros extras dos refuters",
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        merged_estimate_params = {**estimate_params, **estimate_extra}
        merged_refuter_params = {**refuter_params}
        for ref_name, ref_cfg in refuter_extra.items():
            current = merged_refuter_params.get(ref_name, {})
            if isinstance(ref_cfg, dict):
                merged_refuter_params[ref_name] = {**current, **ref_cfg}

        progress_widget = progress_bar_area.progress(
            0, text="Iniciando inferência causal...",
        )
        progress_messages: list = []
        started_at = monotonic()

        def _ui_progress(message: str, progress=None) -> None:
            elapsed = monotonic() - started_at
            if progress is not None:
                pct = int(max(0.0, min(1.0, float(progress))) * 100)
                progress_widget.progress(
                    pct,
                    text=(
                        "Inferência causal em andamento... "
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
            result = run_inference(
                InferenceRunConfig(
                    dataset_path=dataset_path,
                    graph_path=graph_path,
                    estimate_method=estimate_method,
                    estimate_params=merged_estimate_params,
                    refuters=refuters,
                    refuter_params=merged_refuter_params,
                    treatment_columns=effective_treatments,
                    outcome_columns=effective_outcomes,
                    all_pairs=all_pairs,
                    paths=paths,
                ),
                progress_callback=_ui_progress,
            )
            total_seconds = monotonic() - started_at
            progress_widget.progress(
                100,
                text=f"Inferência concluída. Tempo total: {_format_elapsed(total_seconds)}",
            )
            st.session_state.inference_result = result
            n_pairs = len(result.get("pairs", []))
            n_skipped = len(result.get("skipped", []))
            if n_pairs == 0 and n_skipped == 0:
                st.warning(
                    "Inferência rodou mas não avaliou nenhum par. Verifique "
                    "as seleções de treatment/outcome acima."
                )
            else:
                st.success(
                    f"Inferência concluída: {n_pairs} par(es) com sucesso, "
                    f"{n_skipped} pulado(s)."
                )
        except Exception as exc:
            st.error(
                f"Falha na inferência causal: {type(exc).__name__}: {exc}"
            )
            with st.expander("Detalhes técnicos do erro"):
                import traceback as _tb
                st.code(_tb.format_exc(), language="text")

    # ------------------------------------------------------------------
    # 6. Resultados
    # ------------------------------------------------------------------
    result = st.session_state.get("inference_result")
    if not result:
        return

    render_section_header("5. Resultados")

    pairs = result.get("pairs", [])
    skipped = result.get("skipped", [])

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Pares com sucesso", len(pairs))
    with metric_cols[1]:
        st.metric("Pares pulados", len(skipped))
    with metric_cols[2]:
        st.metric("Método", result.get("estimate_method", "-"))

    if pairs:
        st.markdown("#### Tabela-resumo")
        rows = []
        for p in pairs:
            refuters_p = p.get("refuters") or []
            n_total = len(refuters_p)
            n_passed = sum(1 for r in refuters_p if r.get("passes") is True)
            n_failed = sum(1 for r in refuters_p if r.get("passes") is False)
            n_inconclusive = n_total - n_passed - n_failed

            estimate_value = p.get("estimate_value")
            estimate_str = (
                f"{estimate_value:.4f}" if isinstance(estimate_value, (int, float))
                else "—"
            )
            rows.append({
                "Treatment": p.get("treatment", ""),
                "Outcome": p.get("outcome", ""),
                "Efeito estimado": estimate_str,
                "Refuters passou": n_passed,
                "Refuters falhou": n_failed,
                "Refuters inconclusivos": n_inconclusive,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Detalhes por par")
        for p in pairs:
            label = f"{p['treatment']}  →  {p['outcome']}"
            estimate_value = p.get("estimate_value")
            if isinstance(estimate_value, (int, float)):
                label += f"   (efeito={estimate_value:.4f})"
            with st.expander(label, expanded=False):
                _render_estimate_card(p)
    else:
        st.info("Nenhum par foi avaliado com sucesso.")

    if skipped:
        st.markdown("#### Pares pulados")
        st.caption(
            "Pares que não puderam ser avaliados. Veja a coluna 'Motivo' para detalhes."
        )
        skipped_rows = [
            {
                "Treatment": s.get("treatment", ""),
                "Outcome": s.get("outcome", ""),
                "Motivo": s.get("reason", ""),
            }
            for s in skipped
        ]
        st.dataframe(
            pd.DataFrame(skipped_rows),
            use_container_width=True,
            hide_index=True,
        )

    graph_debug = result.get("graph_debug") or {}
    if graph_debug.get("undirected_dropped", 0) or graph_debug.get("pag_dropped", 0):
        with st.expander("Avisos sobre o grafo causal", expanded=False):
            st.write(graph_debug)
            st.caption(
                "O DoWhy requer um DAG completamente direcionado. Arestas "
                "não-direcionadas (de algoritmos como PC, que produzem CPDAG) ou "
                "arestas com ponta circular (de FCI, que produz PAG) são descartadas. "
                "Considere usar FGES no passo de descoberta, que produz DAG completo."
            )

    result_dir = paths.causal_results_dir / "inference"
    if result_dir.exists():
        inference_zip = zip_paths(
            paths.workdir / "causal_inference_outputs.zip",
            [result_dir],
            base_dir=paths.workdir,
        )
        st.download_button(
            "Baixar ZIP da inferência",
            data=inference_zip.read_bytes(),
            file_name=inference_zip.name,
            mime="application/zip",
            key="download_inference_zip",
        )