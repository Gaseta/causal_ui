from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.domain.models import FileSelection, PreprocessingConfig, RuntimePaths
from src.services.archive_service import zip_paths
from src.services.preprocessing_service import prepare_data
from src.ui.common import (
    default_maf_selection,
    guess_column,
    inspect_uploaded_table,
    render_info_card,
    render_section_header,
    save_upload,
)


OPTIONAL_REACTOME_FILENAMES = {
    "drivers": "NCG_cancerdrivers_systemslevelproperties.tsv",
    "geneset": "pathwayGeneSet.pickle",
    "networks": "pathwayNetworks.pickle",
    "gmt": "ReactomePathways.gmt",
}



def _resolve_optional_default(paths: RuntimePaths, filename: str) -> Path | None:
    candidate_workdir = paths.project_root / filename
    if candidate_workdir.exists():
        return candidate_workdir
    candidate_data = paths.data_dir / filename
    if candidate_data.exists():
        return candidate_data
    root_candidate = Path("/mnt/data") / filename
    if root_candidate.exists():
        return root_candidate
    return None



def render(paths: RuntimePaths) -> None:
    st.subheader("Pré-processamento de dados")
    render_section_header(
        "1. Uploads e seleção de colunas",
        "Anexe MAF, clinical sample e clinical patient. A interface lista as colunas e permite controlar o pipeline sem depender de config.py.",
    )

    left, right = st.columns(2)
    with left:
        maf_file = st.file_uploader("Arquivo MAF / mutações", type=["txt", "tsv", "csv"], key="ui_maf_upload")
        sample_file = st.file_uploader("Arquivo clínico de sample", type=["txt", "tsv", "csv"], key="ui_sample_upload")
        patient_file = st.file_uploader("Arquivo clínico de patient", type=["txt", "tsv", "csv"], key="ui_patient_upload")
    with right:
        pathway_gmt_file = st.file_uploader("Reactome GMT (opcional)", type=["gmt", "txt"], key="ui_gmt_upload")
        drivers_file = st.file_uploader("Tabela de drivers / NCG (opcional)", type=["tsv", "txt", "csv"], key="ui_drivers_upload")
        geneset_file = st.file_uploader("pathwayGeneSet.pickle (opcional)", type=["pickle", "pkl"], key="ui_geneset_upload")
        networks_file = st.file_uploader("pathwayNetworks.pickle (opcional)", type=["pickle", "pkl"], key="ui_networks_upload")

    if not (maf_file and sample_file and patient_file):
        st.info("Envie os três arquivos principais para habilitar a configuração do pré-processamento.")
        return

    maf_path = save_upload(maf_file, "maf", paths)
    sample_path = save_upload(sample_file, "sample", paths)
    patient_path = save_upload(patient_file, "patient", paths)

    maf_preview, maf_cols = inspect_uploaded_table(str(maf_path))
    sample_preview, sample_cols = inspect_uploaded_table(str(sample_path))
    patient_preview, patient_cols = inspect_uploaded_table(str(patient_path))

    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.markdown("**MAF**")
        st.dataframe(maf_preview, use_container_width=True, height=220)
    with preview_cols[1]:
        st.markdown("**Sample**")
        st.dataframe(sample_preview, use_container_width=True, height=220)
    with preview_cols[2]:
        st.markdown("**Patient**")
        st.dataframe(patient_preview, use_container_width=True, height=220)

    render_section_header("2. Seleção estrutural")
    select_cols = st.columns(3)
    with select_cols[0]:
        maf_selected = st.multiselect("Colunas do MAF", maf_cols, default=default_maf_selection(maf_cols), key="ui_maf_columns")
    with select_cols[1]:
        sample_defaults = [c for c in [guess_column(sample_cols, "patient"), guess_column(sample_cols, "sample")] if c]
        sample_selected = st.multiselect("Colunas do sample", sample_cols, default=sample_defaults, key="ui_sample_columns")
    with select_cols[2]:
        patient_defaults = [c for c in [guess_column(patient_cols, "patient")] if c]
        patient_selected = st.multiselect("Colunas do patient", patient_cols, default=patient_defaults, key="ui_patient_columns")

    patient_options = sorted({*sample_cols, *patient_cols})
    structural_cols = st.columns(4)
    with structural_cols[0]:
        patient_id_col = st.selectbox("Coluna de paciente", options=patient_options, index=patient_options.index(guess_column(patient_options, "patient")) if guess_column(patient_options, "patient") in patient_options else 0)
    with structural_cols[1]:
        sample_id_guess = guess_column(sample_cols, "sample")
        sample_id_col = st.selectbox("Coluna de sample", options=sample_cols, index=sample_cols.index(sample_id_guess) if sample_id_guess in sample_cols else 0)
    with structural_cols[2]:
        maf_gene_guess = guess_column(maf_cols, "gene")
        maf_gene_col = st.selectbox("Coluna de gene no MAF", options=maf_cols, index=maf_cols.index(maf_gene_guess) if maf_gene_guess in maf_cols else 0)
    with structural_cols[3]:
        maf_sample_guess = guess_column(maf_cols, "sample")
        maf_sample_col = st.selectbox("Coluna de sample no MAF", options=maf_cols, index=maf_cols.index(maf_sample_guess) if maf_sample_guess in maf_cols else 0)

    render_section_header("3. Regras e transformações")
    options_cols = st.columns(4)
    with options_cols[0]:
        analyze_subpathways = st.checkbox("Incluir subpathways", value=True)
        use_all_pathways = st.checkbox("Usar todos os pathways disponíveis", value=True)
    with options_cols[1]:
        convert_outcome = st.checkbox("Criar outcome binário", value=False)
        outcome_regex = st.text_input("Regex outcome positivo", value=r"DECEASED|DEAD|DIED|1:DECEASED")
    with options_cols[2]:
        convert_stage = st.checkbox("Converter estágio romano", value=False)
        stage_regex = st.text_input("Regex estágio romano", value=r"\b(I{1,3}|IV|V)\b")
    with options_cols[3]:
        subgroup_min_fraction = st.number_input("Fração mínima por pathway", value=0.10, min_value=0.0, max_value=1.0, step=0.01)
        fisher_p = st.number_input("Fisher p-value", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        fisher_or = st.number_input("Fisher odds ratio", value=1.0, min_value=0.0, step=0.1)

    super_pathways_text = st.text_area(
        "Superpathways específicos (um por linha, opcional)",
        value="",
        help="Deixe em branco para usar todos. Se preencher, a análise restringe aos roots informados.",
    )

    binary_patient_cols = st.multiselect(
        "Colunas de patient para outcome binário",
        options=[c for c in patient_selected if c != patient_id_col],
        default=[],
        disabled=not convert_outcome,
    )
    binary_sample_cols = st.multiselect(
        "Colunas de sample para outcome binário",
        options=[c for c in sample_selected if c not in {patient_id_col, sample_id_col}],
        default=[],
        disabled=not convert_outcome,
    )
    stage_patient_cols = st.multiselect(
        "Colunas de patient para estágio romano",
        options=[c for c in patient_selected if c != patient_id_col],
        default=[],
        disabled=not convert_stage,
    )
    stage_sample_cols = st.multiselect(
        "Colunas de sample para estágio romano",
        options=[c for c in sample_selected if c not in {patient_id_col, sample_id_col}],
        default=[],
        disabled=not convert_stage,
    )

    render_info_card(
        "Arquivos opcionais detectados",
        [
            f"Drivers padrão: {_resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES['drivers'])}",
            f"Geneset padrão: {_resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES['geneset'])}",
            f"Networks padrão: {_resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES['networks'])}",
            f"GMT padrão: {_resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES['gmt'])}",
        ],
    )

    if st.button("Executar pré-processamento", type="primary", key="run_preprocessing"):
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
            pathway_gmt_file=save_upload(pathway_gmt_file, "gmt", paths) if pathway_gmt_file else _resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES["gmt"]),
            drivers_file=save_upload(drivers_file, "drivers", paths) if drivers_file else _resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES["drivers"]),
            pathway_geneset_pickle=save_upload(geneset_file, "geneset", paths) if geneset_file else _resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES["geneset"]),
            pathway_networks_pickle=save_upload(networks_file, "networks", paths) if networks_file else _resolve_optional_default(paths, OPTIONAL_REACTOME_FILENAMES["networks"]),
            selection=selection,
            analyze_subpathways=analyze_subpathways,
            subgroup_min_fraction=float(subgroup_min_fraction),
            fisher_pvalue_threshold=float(fisher_p),
            fisher_or_threshold=float(fisher_or),
            convert_outcome_to_binary=convert_outcome,
            outcome_positive_regex=outcome_regex,
            outcome_binary_columns_sample=binary_sample_cols,
            outcome_binary_columns_patient=binary_patient_cols,
            convert_stage_to_numeric=convert_stage,
            stage_roman_regex=stage_regex,
            stage_roman_columns_sample=stage_sample_cols,
            stage_roman_columns_patient=stage_patient_cols,
            super_pathways_names=super_pathways,
        )
        with st.spinner("Executando pré-processamento..."):
            result = prepare_data(cfg)
        st.session_state.preprocess_result = result
        st.success("Pré-processamento concluído.")

    result = st.session_state.get("preprocess_result")
    if not result:
        return

    render_section_header("4. Resultados")
    group_resume = result.get("group_resume")
    if group_resume is not None:
        st.dataframe(group_resume, use_container_width=True, height=320)

    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Datasets individuais", len(result.get("dataset_paths", [])))
    with metrics[1]:
        st.metric("Pathways no catálogo", len(result.get("pathway_catalog", [])) if result.get("pathway_catalog") is not None else 0)
    with metrics[2]:
        st.metric("Patients no CombinedDF", 0 if result.get("combined_df") is None else len(result["combined_df"]))
    with metrics[3]:
        st.metric("Datasets com sinal", len(result.get("patient_vs_pathways", {})))

    combined_path = result.get("combined_df_path")
    if combined_path:
        st.code(str(combined_path), language="text")
        if Path(combined_path).exists():
            st.dataframe(result["combined_df"].head(20), use_container_width=True, height=280)

    files_to_zip = [result["datasets_dir"], result["group_resume_path"]]
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
