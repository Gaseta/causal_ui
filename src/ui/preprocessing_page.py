from __future__ import annotations

from pathlib import Path
from time import monotonic

import streamlit as st

from src.domain.models import FileSelection, PreprocessingConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.cbioportal_fetcher import REQUIRED_FILES, fetch_dataset
from src.services.preprocessing_service import prepare_data
from src.ui.common import (
    default_maf_selection,
    guess_column,
    guess_outcome_column,
    inspect_uploaded_table,
    render_section_header,
    save_upload,
)


def _uploaded_file_signature(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    return "|".join(
        [
            str(getattr(uploaded_file, "file_id", "")),
            str(uploaded_file.name),
            str(uploaded_file.size),
            str(uploaded_file.type),
        ]
    )


def _reset_preprocessing_state_if_uploads_changed(maf_file, sample_file, patient_file) -> None:
    current_signature = "::".join(
        [
            _uploaded_file_signature(maf_file),
            _uploaded_file_signature(sample_file),
            _uploaded_file_signature(patient_file),
        ]
    )

    previous_signature = st.session_state.get("preprocessing_upload_signature")

    if previous_signature and previous_signature != current_signature:
        keys_to_clear = [
            "preprocess_result",
            "preprocess_progress_messages",
            "preprocessing_running",
            "ui_maf_columns",
            "ui_sample_columns",
            "ui_patient_columns",
            "run_preprocessing",
        ]

        for key in keys_to_clear:
            st.session_state.pop(key, None)

    st.session_state["preprocessing_upload_signature"] = current_signature


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {remaining_minutes:02d}min {remaining_seconds:02d}s"

    if minutes:
        return f"{minutes}min {remaining_seconds:02d}s"

    return f"{remaining_seconds}s"


def _general_preprocessing_progress(message: str, elapsed_seconds: float, last_progress: int) -> int:
    normalized_message = str(message or "").lower()

    milestone = 5

    if any(term in normalized_message for term in ["inicial", "start", "carreg", "load", "read"]):
        milestone = 10
    elif any(term in normalized_message for term in ["clinical", "sample", "patient", "maf", "mutation"]):
        milestone = 22
    elif any(term in normalized_message for term in ["driver", "gene", "mutation matrix", "pxg"]):
        milestone = 32
    elif any(term in normalized_message for term in ["pathway", "subpathway", "reactome", "deeping"]):
        milestone = 45
    elif any(term in normalized_message for term in ["combined", "merge", "integr", "pxp", "dataset"]):
        milestone = 75
    elif any(term in normalized_message for term in ["saving", "salv", "zip", "export", "write"]):
        milestone = 92
    elif any(term in normalized_message for term in ["final", "conclu", "done", "finish"]):
        milestone = 100

    elapsed_based_progress = min(95, 8 + int(elapsed_seconds / 6))

    return max(last_progress, milestone, elapsed_based_progress)


# Limites para evitar estouro de memória em pipelines longos (Reactome tem
# milhares de pathways). A lista de mensagens da UI fica capada e o painel
# renderiza apenas a janela final, com uma chamada de st.text/markdown em
# vez de um widget Streamlit por linha.
_MAX_HISTORY_MESSAGES = 150  # quantas mensagens guardamos em session_state
_RENDER_TAIL = 60             # quantas mostramos no painel histórico


def _render_progress_panel(container, messages: list[str], current_message: str) -> None:
    """Renderiza o painel de progresso de forma econômica.

    Antes: criava um widget Streamlit (st.success/warning/write) por mensagem,
    o que custa muito em memória quando há centenas/milhares de mensagens.
    Agora: usa UM ÚNICO bloco de texto monoespaçado com a janela final do
    histórico, o que mantém o uso de memória constante independentemente
    de quantas mensagens já passaram.

    O título e o card "Etapa atual" são gerenciados fora dessa função (no
    corpo principal do page), para evitar duplicação visual.
    """
    container.empty()

    tail = messages[-_RENDER_TAIL:]
    if not tail:
        return

    with container.container():
        st.markdown("**Histórico recente:**")
        # Junta tudo num bloco só: barato em memória e em tempo de render.
        st.code("\n".join(tail), language="text")


def _resolve_data_sources(paths: RuntimePaths):
    """Resolve os 3 paths de dados a partir de upload manual ou fetch do cBioPortal.

    Retorna uma tupla ``(maf_path, sample_path, patient_path, signature)`` ou
    ``(None, None, None, "")`` se ainda não houver dados completos.

    O ``signature`` é uma string que muda quando a origem dos dados muda; ela é
    usada para invalidar o estado do pré-processamento se o usuário trocar de
    dataset/arquivos no meio do fluxo.
    """
    tab_upload, tab_cbio = st.tabs(["Upload manual", "Buscar do cBioPortal Datahub"])

    maf_path = sample_path = patient_path = None
    signature = ""

    with tab_upload:
        st.caption(
            "Envie os três arquivos principais a partir do seu computador. "
            "Os arquivos auxiliares de Reactome e drivers são sempre lidos da pasta data/ do projeto."
        )

        maf_file = st.file_uploader("Arquivo MAF / mutações", type=["txt", "tsv", "csv"], key="ui_maf_upload")
        sample_file = st.file_uploader("Arquivo clínico de sample", type=["txt", "tsv", "csv"], key="ui_sample_upload")
        patient_file = st.file_uploader("Arquivo clínico de patient", type=["txt", "tsv", "csv"], key="ui_patient_upload")

        _reset_preprocessing_state_if_uploads_changed(maf_file, sample_file, patient_file)

        if maf_file and sample_file and patient_file:
            maf_path = save_upload(maf_file, "maf", paths)
            sample_path = save_upload(sample_file, "sample", paths)
            patient_path = save_upload(patient_file, "patient", paths)
            signature = "upload::" + "::".join([
                _uploaded_file_signature(maf_file),
                _uploaded_file_signature(sample_file),
                _uploaded_file_signature(patient_file),
            ])

    with tab_cbio:
        st.caption(
            "Digite o identificador de um estudo público do cBioPortal Datahub "
            "(por exemplo: `brca_metabric`, `luad_tcga`, `acc_2019`). "
            "Baixaremos automaticamente os arquivos necessários: "
            f"`{', '.join(REQUIRED_FILES)}`. "
            "Os arquivos ficam em cache local — uma busca subsequente do mesmo dataset não baixa de novo."
        )

        cbio_cols = st.columns([3, 1])

        with cbio_cols[0]:
            dataset_input = st.text_input(
                "Nome do dataset",
                value=st.session_state.get("cbio_dataset_name", ""),
                key="ui_cbio_dataset",
                help=(
                    "Identificador do estudo conforme aparece na URL do datahub. "
                    "Por exemplo, em https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric "
                    "o identificador é 'brca_metabric'."
                ),
                placeholder="brca_metabric",
            )

        with cbio_cols[1]:
            st.markdown("&nbsp;", unsafe_allow_html=True)  # alinha verticalmente
            fetch_clicked = st.button(
                "Baixar do cBioPortal",
                key="ui_cbio_fetch_btn",
                disabled=not bool(dataset_input.strip()),
                width="stretch",
            )

        cbio_log_area = st.empty()
        cbio_status_area = st.empty()

        if fetch_clicked and dataset_input.strip():
            cbio_target_root = paths.workdir / "cbioportal_datasets"
            cbio_messages: list[str] = []

            def _cbio_progress(message: str, progress=None) -> None:
                cbio_messages.append(message)
                # Mantém só as últimas 20 mensagens.
                if len(cbio_messages) > 20:
                    del cbio_messages[: len(cbio_messages) - 20]
                cbio_log_area.code("\n".join(cbio_messages), language="text")

            try:
                with st.spinner(f"Baixando dataset '{dataset_input.strip()}' do cBioPortal..."):
                    fetch_result = fetch_dataset(
                        dataset_name=dataset_input.strip(),
                        target_root=cbio_target_root,
                        progress_callback=_cbio_progress,
                    )

                st.session_state["cbio_dataset_name"] = dataset_input.strip()
                st.session_state["cbio_fetch_result"] = {
                    "dataset": fetch_result.dataset,
                    "mutations": str(fetch_result.mutations_file),
                    "sample": str(fetch_result.clinical_sample_file),
                    "patient": str(fetch_result.clinical_patient_file),
                    "downloaded": fetch_result.files_downloaded,
                    "cached": fetch_result.files_already_present,
                }
                # Invalida o estado anterior do pré-processamento (mudou o dataset).
                for key in [
                    "preprocess_result", "preprocess_progress_messages",
                    "preprocessing_running", "ui_maf_columns",
                    "ui_sample_columns", "ui_patient_columns",
                ]:
                    st.session_state.pop(key, None)

                cbio_status_area.success(
                    f"Dataset '{fetch_result.dataset}' pronto. "
                    f"Baixados: {len(fetch_result.files_downloaded)}; "
                    f"já em cache: {len(fetch_result.files_already_present)}."
                )
            except FileNotFoundError as exc:
                cbio_status_area.error(str(exc))
            except (RuntimeError, ValueError) as exc:
                cbio_status_area.error(str(exc))

        # Reusa o resultado do último fetch armazenado em session_state.
        cbio_state = st.session_state.get("cbio_fetch_result")
        if cbio_state:
            ds_name = cbio_state["dataset"]
            cbio_status_area.info(
                f"Usando dataset do cBioPortal: **{ds_name}**. "
                f"Arquivos em `{Path(cbio_state['mutations']).parent}`. "
                f"Limpe digitando outro nome e clicando em 'Baixar do cBioPortal' para trocar."
            )
            maf_path_cbio = Path(cbio_state["mutations"])
            sample_path_cbio = Path(cbio_state["sample"])
            patient_path_cbio = Path(cbio_state["patient"])

            # Só sobrescreve se ainda não houver upload manual ativo.
            if maf_path is None:
                maf_path = maf_path_cbio
                sample_path = sample_path_cbio
                patient_path = patient_path_cbio
                signature = f"cbio::{ds_name}"

    return maf_path, sample_path, patient_path, signature


def render(paths: RuntimePaths) -> None:
    st.subheader("Pré-processamento de dados")
    render_section_header(
        "1. Uploads e seleção de colunas",
        "Envie os arquivos ou busque um estudo público do cBioPortal Datahub. "
        "Os arquivos auxiliares de Reactome e drivers são sempre lidos da pasta data/ do projeto.",
    )

    maf_path, sample_path, patient_path, data_signature = _resolve_data_sources(paths)

    if not (maf_path and sample_path and patient_path):
        st.info(
            "Envie os três arquivos pela aba 'Upload manual' "
            "ou busque um estudo pela aba 'Buscar do cBioPortal Datahub' "
            "para habilitar a configuração do pré-processamento."
        )
        return

    # Invalida o estado se a origem dos dados mudou desde a última execução.
    previous_data_signature = st.session_state.get("preprocessing_data_signature")
    if previous_data_signature and previous_data_signature != data_signature:
        for key in [
            "preprocess_result", "preprocess_progress_messages",
            "preprocessing_running", "ui_maf_columns",
            "ui_sample_columns", "ui_patient_columns", "run_preprocessing",
        ]:
            st.session_state.pop(key, None)
    st.session_state["preprocessing_data_signature"] = data_signature

    maf_preview, maf_cols = inspect_uploaded_table(str(maf_path))
    sample_preview, sample_cols = inspect_uploaded_table(str(sample_path))
    patient_preview, patient_cols = inspect_uploaded_table(str(patient_path))

    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.markdown("**MAF**")
        st.dataframe(maf_preview, width='stretch', height=220)
    with preview_cols[1]:
        st.markdown("**Sample**")
        st.dataframe(sample_preview, width='stretch', height=220)
    with preview_cols[2]:
        st.markdown("**Patient**")
        st.dataframe(patient_preview, width='stretch', height=220)

    render_section_header("2. Seleção estrutural")
    select_cols = st.columns(3)

    with select_cols[0]:
        maf_selected = st.multiselect(
            "Colunas do MAF",
            maf_cols,
            default=default_maf_selection(maf_cols),
            key="ui_maf_columns",
            help=(
                "Selecione somente as colunas necessárias do arquivo MAF. "
                "Por padrão, são usadas as colunas essenciais para identificar genes, localização genômica, "
                "tipo de variante e amostra tumoral."
            ),
        )

    with select_cols[1]:
        _tmb_col = next(
            (c for c in sample_cols if str(c).strip().upper() == "TMB_NONSYNONYMOUS"),
            None,
        )
        sample_defaults = [
            c
            for c in [
                guess_column(sample_cols, "patient"),
                guess_column(sample_cols, "sample"),
                _tmb_col,
            ]
            if c
        ]

        sample_selected = st.multiselect(
            "Colunas do sample",
            sample_cols,
            default=sample_defaults,
            key="ui_sample_columns",
            help=(
                "Selecione as colunas do arquivo clínico de amostra que devem ser preservadas no dataset final. "
                "Esse arquivo normalmente contém informações associadas à amostra tumoral, como TMB, tipo de amostra, "
                "identificador de paciente e identificador de sample."
            ),
        )

    with select_cols[2]:
        _os_status_col = next(
            (c for c in patient_cols if str(c).strip().upper() == "OS_STATUS"),
            None,
        )
        patient_defaults = [
            c
            for c in [
                guess_column(patient_cols, "patient"),
                _os_status_col,
            ]
            if c
        ]

        patient_selected = st.multiselect(
            "Colunas do patient",
            patient_cols,
            default=patient_defaults,
            key="ui_patient_columns",
            help=(
                "Selecione as colunas clínicas em nível de paciente que devem ser preservadas no dataset final. "
                "Esse arquivo normalmente contém idade, sexo, status clínico, estágio, sobrevida ou outras variáveis clínicas."
            ),
        )

    patient_options = sorted({*sample_cols, *patient_cols})
    structural_cols = st.columns(4)

    with structural_cols[0]:
        patient_guess = guess_column(patient_options, "patient")
        patient_id_col = st.selectbox(
            "Coluna de paciente",
            options=patient_options,
            index=patient_options.index(patient_guess) if patient_guess in patient_options else 0,
            help=(
                "Coluna que identifica unicamente cada paciente. "
                "Ela é usada para integrar dados clínicos de patient, dados de sample e mutações do MAF."
            ),
        )

    with structural_cols[1]:
        sample_id_guess = guess_column(sample_cols, "sample")
        sample_id_col = st.selectbox(
            "Coluna de sample",
            options=sample_cols,
            index=sample_cols.index(sample_id_guess) if sample_id_guess in sample_cols else 0,
            help=(
                "Coluna que identifica unicamente cada amostra tumoral no arquivo de sample. "
                "Ela deve ser compatível com o identificador de amostra presente no MAF."
            ),
        )

    with structural_cols[2]:
        maf_gene_guess = guess_column(maf_cols, "gene")
        maf_gene_col = st.selectbox(
            "Coluna de gene no MAF",
            options=maf_cols,
            index=maf_cols.index(maf_gene_guess) if maf_gene_guess in maf_cols else 0,
            help=(
                "Coluna do MAF que contém o símbolo oficial do gene afetado pela mutação. "
                "Normalmente corresponde a Hugo_Symbol."
            ),
        )

    with structural_cols[3]:
        maf_sample_guess = guess_column(maf_cols, "sample")
        maf_sample_col = st.selectbox(
            "Coluna de sample no MAF",
            options=maf_cols,
            index=maf_cols.index(maf_sample_guess) if maf_sample_guess in maf_cols else 0,
            help=(
                "Coluna do MAF que contém o identificador da amostra tumoral. "
                "Normalmente corresponde a Tumor_Sample_Barcode."
            ),
        )

    render_section_header(
        "3. Regras de análise",
        "Defina o outcome principal e os critérios usados para identificar pathways relevantes.",
    )

    available_outcome_columns = [
        c
        for c in sorted({*sample_selected, *patient_selected})
        if c not in {patient_id_col, sample_id_col}
    ]

    if not available_outcome_columns:
        st.error("Selecione ao menos uma coluna clínica de patient ou sample para definir o outcome.")
        st.stop()

    preferred_outcomes = [
        "OS_STATUS",
        "OS Status",
        "os_status",
        "os status",
    ]

    default_outcome = None

    for pref in preferred_outcomes:
        for col in available_outcome_columns:
            if str(col).strip().lower().replace(" ", "_") == pref.lower().replace(" ", "_"):
                default_outcome = col
                break
        if default_outcome:
            break

    if not default_outcome:
        default_outcome = guess_outcome_column(available_outcome_columns)

    default_outcome_index = (
        available_outcome_columns.index(default_outcome)
        if default_outcome in available_outcome_columns
        else 0
    )

    outcome_col = st.selectbox(
        "Coluna de outcome",
        options=available_outcome_columns,
        index=default_outcome_index,
        help=(
            "Selecione a coluna que representa o desfecho clínico principal da análise, "
            "como status de sobrevida, vital status ou OS_STATUS. "
            "Se a coluna for textual, o sistema tentará criar automaticamente uma versão binária."
        ),
    )

    outcome_source = (
        "sample"
        if outcome_col in sample_selected and outcome_col not in patient_selected
        else "patient"
    )

    st.caption(
        f"Outcome selecionado: `{outcome_col}` "
        f"({'arquivo sample' if outcome_source == 'sample' else 'arquivo patient'})"
    )

    options_cols = st.columns(3)

    with options_cols[0]:
        analyze_subpathways = st.checkbox(
            "Incluir subpathways",
            value=True,
            help=(
                "Quando habilitado, além dos pathways principais, o pré-processamento também avalia os subpathways "
                "presentes na hierarquia do Reactome carregada a partir da pasta data/."
            ),
        )

        use_all_pathways = st.checkbox(
            "Usar todos os pathways disponíveis",
            value=True,
            help=(
                "Quando habilitado, todos os pathways encontrados nos arquivos de referência da pasta data/ são avaliados. "
                "Quando desabilitado, você pode restringir a análise aos superpathways informados no campo abaixo."
            ),
        )

    with options_cols[1]:
        subgroup_min_fraction = st.number_input(
            "Fração mínima por pathway",
            value=0.20,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help=(
                "Fração mínima de pacientes que devem apresentar sinal em um pathway para que ele seja mantido. "
                "Exemplo: 0.20 significa que o pathway precisa estar presente em pelo menos 20% dos pacientes avaliados."
            ),
        )

    super_pathways_text = st.text_area(
        "Superpathways específicos (um por linha, opcional)",
        value="",
        disabled=use_all_pathways,
        help=(
            "Use este campo apenas se quiser restringir a análise a alguns superpathways específicos. "
            "Deixe vazio, ou mantenha 'Usar todos os pathways disponíveis' habilitado, para avaliar todos os pathways."
        ),
    )

    # ----------------------------------------------------------------
    # Estratégia de preenchimento dos valores das colunas de pathway
    # ----------------------------------------------------------------
    st.markdown("##### Estratégia de preenchimento dos valores de pathway")

    strategy_options = [
        "Seleção por teste de Fisher, valores com máscara estatística",
        "Seleção por teste de Fisher, valores densos por contagem direta",
        "Seleção por cobertura mínima, valores densos por contagem direta",
    ]

    strategy_captions = [
        "Para cada paciente, faz um teste de Fisher entre os genes mutados e os genes do pathway, "
        "usando como universo o conjunto de genes presentes nas mutações. "
        "O pathway é mantido quando uma fração mínima dos pacientes apresenta OR > 1 e p-valor < 0.05. "
        "No resultado final, só os pacientes que passam o teste recebem um valor numérico (a contagem "
        "de drivers no pathway); os demais ficam como NaN. "
        "Vantajoso quando o objetivo é uma análise estatística tradicional, em que o conceito de "
        "enriquecimento significativo guia a interpretação.",

        "Aplica o mesmo critério de seleção de pathway descrito acima (Fisher com OR > 1 e p-valor < 0.05 "
        "atingindo a fração mínima de pacientes), mas preenche o resultado final de forma densa: TODOS os "
        "pacientes recebem um valor numérico — a contagem de drivers do pathway — e quem não tem nenhuma "
        "mutação no pathway recebe zero, em vez de NaN. "
        "Vantajoso quando o resultado será consumido por algoritmos que assumem dados completos e sem "
        "viés de seleção (por exemplo, descoberta causal, regressões, métodos de clustering ou redes neurais), "
        "porque a ausência de máscara elimina dependência condicional entre o valor e sua significância.",

        "Não usa teste estatístico. Um pathway é mantido se a fração de pacientes com pelo menos uma mutação "
        "no pathway (cobertura) atinge um limiar configurável. O resultado final é preenchido densamente: "
        "todos os pacientes recebem a contagem direta de drivers no pathway, com zero quando não há mutação. "
        "Vantajoso quando o conjunto de genes observados nas mutações é pequeno (por exemplo, em ensaios que "
        "sequenciam apenas um painel de genes pré-selecionado). Nesses casos, o teste de Fisher torna-se "
        "matematicamente desfavorável porque o universo de comparação é reduzido demais — o critério de "
        "cobertura permanece estável e dá resultados interpretáveis.",
    ]

    strategy_label_to_key = {
        strategy_options[0]: "fisher_masked",
        strategy_options[1]: "fisher_dense",
        strategy_options[2]: "coverage_dense",
    }

    selected_strategy_label = st.radio(
        "Como preencher os valores das colunas de pathway?",
        options=strategy_options,
        index=1,  # default = fisher_dense (recomendado para uso geral)
        captions=strategy_captions,
        horizontal=False,
        help=(
            "Define dois aspectos do resultado final: (1) quais pathways aparecem nas colunas do "
            "dataset combinado e (2) como cada célula (paciente × pathway) é preenchida. "
            "Em todas as opções, o valor numérico representa o número de genes drivers mutados do "
            "paciente que pertencem ao pathway — o que muda é o critério de seleção dos pathways "
            "e se as células dos pacientes sem enriquecimento ficam vazias (NaN) ou são preenchidas com zero."
        ),
    )

    pathway_value_strategy_key = strategy_label_to_key[selected_strategy_label]

    # Threshold de cobertura, exibido apenas quando coverage_dense.
    if pathway_value_strategy_key == "coverage_dense":
        coverage_min_fraction = st.number_input(
            "Cobertura mínima por pathway",
            value=0.80,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help=(
                "Fração mínima de pacientes que devem apresentar pelo menos uma mutação driver no "
                "pathway para que ele seja mantido no resultado final. "
                "Exemplo: 0.80 significa que pelo menos 80% dos pacientes precisam ter alguma "
                "mutação no pathway. Em painéis de genes pequenos, valores entre 0.70 e 0.85 "
                "costumam produzir resultados estáveis."
            ),
        )
    else:
        coverage_min_fraction = 0.80  # default; só é usado se a estratégia mudar

    st.markdown("### Andamento do pré-processamento")
    progress_bar = st.empty()
    progress_log = st.empty()

    is_running = st.session_state.get("preprocessing_running", False)

    run_clicked = st.button(
        "Executar pré-processamento",
        type="primary",
        key="run_preprocessing",
        disabled=is_running,
    )

    if is_running:
        st.info("O pré-processamento está em andamento. Aguarde a conclusão antes de executar novamente.")

    if run_clicked:
        st.session_state["preprocessing_running"] = True

        selection = FileSelection(
            maf_columns=maf_selected,
            sample_columns=sample_selected,
            patient_columns=patient_selected,
            patient_id_col=patient_id_col,
            sample_id_col=sample_id_col,
            maf_gene_col=maf_gene_col,
            maf_sample_col=maf_sample_col,
        )

        super_pathways = [line.strip() for line in super_pathways_text.splitlines() if line.strip()] or None
        if use_all_pathways:
            super_pathways = None

        cfg = PreprocessingConfig(
            paths=paths,
            mutations_file=maf_path,
            clinical_sample_file=sample_path,
            clinical_patient_file=patient_path,
            selection=selection,
            analyze_subpathways=analyze_subpathways,
            subgroup_min_fraction=float(subgroup_min_fraction),
            convert_outcome_to_binary=True,
            outcome_binary_columns_sample=[outcome_col] if outcome_source == "sample" else [],
            outcome_binary_columns_patient=[outcome_col] if outcome_source == "patient" else [],
            outcome_positive_regex="__auto__",
            convert_stage_to_numeric=False,
            stage_roman_columns_sample=[],
            stage_roman_columns_patient=[],
            stage_roman_regex="",
            super_pathways_names=super_pathways,
            pathway_value_strategy=pathway_value_strategy_key,
            coverage_min_fraction=float(coverage_min_fraction),
        )

        progress_messages: list[str] = []
        started_at = monotonic()
        last_global_progress = 0

        progress_widget = progress_bar.progress(
            0,
            text="Pré-processamento em andamento... tempo decorrido: 0s",
        )

        _render_progress_panel(
            progress_log,
            ["Inicializando pré-processamento..."],
            "Inicializando pré-processamento...",
        )

        def _ui_progress(message: str, progress: float | None = None) -> None:
            nonlocal last_global_progress

            elapsed_seconds = monotonic() - started_at

            # Atualização de progresso numérico — sempre quando vier.
            if progress is not None:
                normalized_progress = max(0.0, min(1.0, float(progress)))
                # Mapeia o progresso interno para uma faixa global da barra.
                callback_progress = int(5 + normalized_progress * 90)
            elif message:
                callback_progress = _general_preprocessing_progress(
                    message=message,
                    elapsed_seconds=elapsed_seconds,
                    last_progress=last_global_progress,
                )
            else:
                # Mensagem vazia e progresso None: nada a fazer.
                return

            last_global_progress = max(last_global_progress, min(callback_progress, 95))

            progress_widget.progress(
                last_global_progress,
                text=(
                    "Pré-processamento em andamento... "
                    f"tempo decorrido: {_format_elapsed(elapsed_seconds)}"
                ),
            )

            # Mensagem vazia (vinda do _emit_quiet ou de progresso puro):
            # apenas atualiza a barra acima, NÃO acrescenta nada no painel.
            # Isso é o que evita estourar memória durante o loop de pathways.
            if not message:
                return

            # Buffer circular: mantém só as últimas _MAX_HISTORY_MESSAGES.
            progress_messages.append(message)
            if len(progress_messages) > _MAX_HISTORY_MESSAGES:
                # Remove do início em bloco (mais eficiente que pop(0) em loop).
                del progress_messages[: len(progress_messages) - _MAX_HISTORY_MESSAGES]

            _render_progress_panel(progress_log, progress_messages, message)

            st.session_state["preprocess_progress_messages"] = progress_messages

        try:
            _ui_progress("Inicializando pré-processamento...", 0.0)
            result = prepare_data(cfg, progress_callback=_ui_progress)

            st.session_state.preprocess_result = result

            total_seconds = monotonic() - started_at
            final_message = (
                f"Pré-processamento finalizado com sucesso. "
                f"Tempo total: {_format_elapsed(total_seconds)}."
            )

            progress_widget.progress(
                100,
                text=f"Pré-processamento finalizado. Tempo total: {_format_elapsed(total_seconds)}",
            )

            progress_messages.append(final_message)
            st.session_state["preprocess_progress_messages"] = progress_messages

            _render_progress_panel(
                progress_log,
                progress_messages,
                "Pré-processamento finalizado.",
            )

        except Exception as exc:
            total_seconds = monotonic() - started_at
            error_message = f"Erro: {exc}. Tempo até a falha: {_format_elapsed(total_seconds)}."

            progress_messages.append(error_message)
            st.session_state["preprocess_progress_messages"] = progress_messages

            _render_progress_panel(
                progress_log,
                progress_messages,
                "Execução interrompida.",
            )

            raise

        finally:
            st.session_state["preprocessing_running"] = False

    result = st.session_state.get("preprocess_result")
    if not result:
        return

    render_section_header("4. Resultados")
    group_resume = result.get("group_resume")
    if group_resume is not None:
        st.dataframe(group_resume, width='stretch', height=320)

    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Datasets individuais", len(result.get("dataset_paths", [])))
    with metrics[1]:
        st.metric(
            "Pathways no catálogo",
            len(result.get("pathway_catalog", [])) if result.get("pathway_catalog") is not None else 0,
        )
    with metrics[2]:
        st.metric("Patients no CombinedDF", 0 if result.get("combined_df") is None else len(result["combined_df"]))
    with metrics[3]:
        st.metric("Datasets com sinal", len(result.get("patient_vs_pathways", {})))

    execution_messages = st.session_state.get("preprocess_progress_messages") or []
    if execution_messages:
        with st.expander("Resumo da execução do pré-processamento", expanded=False):
            # Bloco único em vez de N widgets — economia massiva de memória
            # quando há muitas mensagens.
            st.code("\n".join(execution_messages), language="text")

    combined_path = result.get("combined_df_path")
    if combined_path:
        st.code(str(combined_path), language="text")
        if Path(combined_path).exists():
            st.dataframe(result["combined_df"].head(20), width='stretch', height=280)

    files_to_zip = [result["datasets_dir"], result["group_resume_path"]]

    patient_vs_pathways_path = result.get("patient_vs_pathways_path")
    if patient_vs_pathways_path:
        files_to_zip.append(patient_vs_pathways_path)

    if combined_path:
        files_to_zip.append(combined_path)

    preprocess_zip = zip_paths(paths.workdir / "preprocessing_outputs.zip", files_to_zip, base_dir=paths.workdir)

    st.download_button(
        "Baixar ZIP do pré-processamento",
        data=preprocess_zip.read_bytes(),
        file_name=preprocess_zip.name,
        mime="application/zip",
        key="download_preprocess_zip",
    )