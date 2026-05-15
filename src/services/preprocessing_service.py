from __future__ import annotations

import pickle
import re
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.domain.models import PreprocessingConfig, RuntimePaths
from src.services.io_utils import ensure_required_columns, normalize_columns, read_table, safe_filename


DEFAULT_SUPER_PATHWAYS = [
    "Autophagy",
    "Cell Cycle",
    "Cell-Cell communication",
    "Cellular responses to stimuli",
    "Chromatin organization",
    "Developmental Biology",
    "DNA Repair",
    "DNA Replication",
    "Extracellular matrix organization",
    "Gene expression (Transcription)",
    "Immune System",
    "Metabolism",
    "Metabolism of proteins",
    "Neuronal System",
    "Programmed Cell Death",
    "Signal Transduction",
    "Transport of small molecules",
    "Vesicle-mediated transport",
]

# Estratégias para preencher os valores das colunas de pathway no CombinedDF.
#
# - "fisher_masked": semântica do notebook original.
#       Seleção de pathway: teste de Fisher (OR>1 AND p<0.05) + threshold X%.
#       Valor da célula: `both` apenas para pacientes que passaram o Fisher;
#                        os demais ficam NaN.
#       Vantagem: reproduz o notebook bit-a-bit.
#       Desvantagem: máscara seletiva → viés MAR/MNAR em descoberta causal.
#
# - "fisher_dense": MESMA seleção de pathway, mas COMBINEDDF DENSO.
#       Seleção de pathway: idêntica a fisher_masked.
#       Valor da célula: `both` para TODOS os pacientes elegíveis;
#                        0 quando o paciente não tem driver naquele pathway.
#       Vantagem: sem viés de seleção, ideal para descoberta causal em
#       datasets whole-exome.
#
# - "coverage_dense": estilo enhanced — sem teste de Fisher.
#       Seleção de pathway: pacientes_com_both>0 / total >= coverage_min.
#       Iteração: todos os pathways do GMT/pathwayGeneSet, sem BFS.
#       Valor da célula: `both` para TODOS os pacientes elegíveis;
#                        0 quando o paciente não tem driver naquele pathway.
#       Vantagem: funciona em painel direcionado (METABRIC).
PATHWAY_VALUE_STRATEGIES = ("fisher_masked", "fisher_dense", "coverage_dense")

REFERENCE_FILENAMES = {
    "drivers": "NCG_cancerdrivers_systemslevelproperties.tsv",
    "geneset": "pathwayGeneSet.pickle",
    "networks": "pathwayNetworks.pickle",
    "gmt": "ReactomePathways.gmt",
}

ProgressCallback = Optional[Callable[[str, Optional[float]], None]]


def _log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def _emit(progress_callback: ProgressCallback, msg: str, progress: Optional[float] = None) -> None:
    """Emite uma mensagem para o callback de UI E para o stdout do servidor.

    Use apenas para MARCOS importantes que o usuário precisa ver na tela
    (início de fase, conclusão, SAVING de pathway, avisos críticos). Cada
    chamada cresce a lista de mensagens da UI, que tem custo de memória
    e de renderização proporcional ao total.
    """
    _log(msg)
    if progress_callback is not None:
        progress_callback(msg, progress)


def _emit_quiet(msg: str, progress_callback: ProgressCallback = None, progress: Optional[float] = None) -> None:
    """Variante "silenciosa": loga só no stdout, NÃO chama o callback de UI.

    Use para mensagens de debug, contagens parciais, iteração por pathway
    e qualquer log de alta frequência. O stdout do servidor é barato; a
    lista de mensagens da UI Streamlit não é.

    O parâmetro ``progress`` é aceito por compatibilidade mas só será
    propagado ao callback se ``progress_callback`` for fornecido E o
    progresso for um marco numérico significativo (ou seja, vem com
    progress != None) — assim a barra de progresso pode avançar sem
    poluir o histórico textual.
    """
    _log(msg)
    if progress_callback is not None and progress is not None:
        # Mensagem vazia evita acumular texto na UI mas permite atualizar a barra.
        progress_callback("", progress)


def _detect_skiprows_by_comment(path: Path, comment_char: str = "#") -> int:
    """Detecta quantas linhas iniciais devem ser puladas em um arquivo tabular.

    Conta as linhas iniciais que começam com ``comment_char`` (default ``#``).
    Esse padrão é robusto para arquivos do cBioPortal/TCGA, que costumam ter
    cabeçalhos de metadados antes da linha real de colunas (por exemplo, o
    MAF do METABRIC abre com ``#Sequenced_Samples: ...`` e arquivos clínicos
    têm até 5 linhas de comentário antes do header verdadeiro).
    """
    skip = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(comment_char):
                    skip += 1
                else:
                    break
    except Exception:
        return 0
    return skip


def resolve_reference_file(paths: RuntimePaths, key: str) -> Optional[Path]:
    filename = REFERENCE_FILENAMES[key]
    candidates = [
        paths.data_dir / filename,
        paths.project_root / "data" / filename,
        paths.project_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _apply_default_reference_files(cfg: PreprocessingConfig) -> Dict[str, Optional[Path]]:
    refs = {
        "drivers": resolve_reference_file(cfg.paths, "drivers"),
        "geneset": resolve_reference_file(cfg.paths, "geneset"),
        "networks": resolve_reference_file(cfg.paths, "networks"),
        "gmt": resolve_reference_file(cfg.paths, "gmt"),
    }
    cfg.drivers_file = refs["drivers"]
    cfg.pathway_geneset_pickle = refs["geneset"]
    cfg.pathway_networks_pickle = refs["networks"]
    cfg.pathway_gmt_file = refs["gmt"]
    return refs


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, float) and np.isnan(value):
        return pd.NA
    # Preserva tipos numéricos (int, float, numpy/pandas Int64) e booleanos.
    # Isso é importante para colunas binarizadas (OS_STATUS -> 0/1) e
    # discretizadas que mantêm representação categórica/numérica original.
    # Sem isso, o _series_to_clean_dict converte tudo para string e o
    # CombinedDF acaba com valores '0'/'1' em vez de 0/1.
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return value
    if isinstance(value, float):  # já tratamos NaN acima
        return value
    # pandas NA é preservado
    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass
    text = str(value).strip()
    if text in {"", "NA", "N/A", "<NA>", "nan", "None"}:
        return pd.NA
    return text


def _normalize_gene_symbol(value: Any) -> str:
    """Normalize gene symbols from MAF, driver lists and Reactome references.

    The notebook compares symbols as plain HGNC-like strings. In the UI flow,
    however, inputs may come from different files and may contain wrappers such
    as ``TP53|7157``, version suffixes, comma-separated aliases or blank markers.
    This function keeps the comparison conservative while removing common noise
    that otherwise makes every patient/pathway intersection equal to zero.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().upper()

    if text in {"", "NA", "N/A", "NAN", "NONE", "NULL", "-", "?"}:
        return ""

    # Common exports may encode gene plus identifier, for example TP53|7157.
    # The first field is usually the gene symbol used by Reactome/HGNC.
    for separator in ["|", ";", ","]:
        if separator in text:
            text = text.split(separator, 1)[0].strip()

    # Remove wrapping quotes and excess inner whitespace.
    text = text.strip("'\"")
    text = re.sub(r"\s+", "", text)

    # Some annotation exports include version suffixes for Ensembl IDs.
    # HGNC symbols with dots are rare in these cancer files, so only strip
    # numeric suffixes when they follow a dot.
    text = re.sub(r"\.\d+$", "", text)

    return text


def _normalize_gene_collection(values: Iterable[Any]) -> List[str]:
    return sorted({g for g in (_normalize_gene_symbol(v) for v in values) if g})


def _debug_gene_examples(values: Iterable[Any], limit: int = 12) -> str:
    genes = list(values)
    if not genes:
        return "[]"
    return str(genes[:limit])


def discretize_in_groups(values: pd.Series, number_of_bins: int = 10) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.dropna().empty:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="string")
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx == mn:
        label = f"{round(mn)}-{round(mx)}"
        return pd.Series([label] * len(values), index=values.index, dtype="string")
    bin_range = (mx - mn) / float(number_of_bins)
    bins = [mn]
    for i in range(number_of_bins):
        bins.append(mn + bin_range * (i + 1))
    labels = [f"{round(bins[i])}-{round(bins[i + 1])}" for i in range(len(bins) - 1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype("string")


def _iqr_upper_threshold(values: pd.Series, factor: float) -> Optional[float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return None
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    return float(q3 + factor * (q3 - q1))


def _roman_to_int(roman: str) -> Optional[int]:
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    return mapping.get(str(roman).upper().strip())


def _apply_categorical_remap(df: pd.DataFrame, remap: Dict[str, Dict[str, str]], label: str) -> None:
    for col, mp in (remap or {}).items():
        if col in df.columns and isinstance(mp, dict) and mp:
            df[col] = df[col].replace(mp)
            _log(f"{label}: remapped {col}")


# Valores que tipicamente representam "indefinido" / "não classificado" em
# datasets clínicos do cBioPortal. Linhas com qualquer um desses valores em
# colunas selecionadas pelo usuário são candidatas a remoção quando o usuário
# habilita o filtro agnóstico (drop_undefined_categorical=True).
# A comparação é feita por string normalizada (lower, sem espaços extras).
UNDEFINED_CATEGORICAL_TOKENS: Set[str] = {
    "tx", "nx", "mx",                # stages T/N/M indeterminados
    "unknown", "not available", "not reported", "not specified",
    "not evaluated", "not applicable",
    "na", "n/a", "<na>", "nan", "none", "null",
    "-", "?", "indeterminate", "indeterminado", "desconhecido",
}


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _apply_row_filters(
    df: pd.DataFrame,
    row_filters: Dict[str, List[str]],
    label: str,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Remove linhas onde uma coluna assume valores que o usuário marcou para descartar.

    ``row_filters`` é um dict ``{coluna: [valores_a_remover]}``. A comparação é
    feita por string normalizada (case-insensitive, espaços extras removidos)
    para ser robusta a variações como ``"Tx"`` vs ``"TX"`` vs ``"  Tx "``.

    Colunas inexistentes no dataframe são silenciosamente ignoradas — isso é
    intencional para que a configuração seja agnóstica: o usuário pode definir
    filtros para colunas que existem em alguns datasets mas não em outros.
    """
    out = df
    initial_size = len(out)
    cols_affected: List[str] = []
    for col, raw_values in (row_filters or {}).items():
        if col not in out.columns or not raw_values:
            continue
        target_tokens = {_normalize_token(v) for v in raw_values if _normalize_token(v)}
        if not target_tokens:
            continue
        normalized = out[col].apply(_normalize_token)
        before = len(out)
        out = out[~normalized.isin(target_tokens)].copy()
        if len(out) < before:
            cols_affected.append(col)
            _log(f"{label}: row_filter removed {before - len(out)} row(s) on {col}")
    total_removed = initial_size - len(out)
    if total_removed > 0:
        _emit(
            progress_callback,
            f"{label}: row_filters removeram {total_removed} linha(s) ({initial_size}→{len(out)}).",
            None,
        )
    return out


def _apply_drop_undefined(
    df: pd.DataFrame,
    candidate_cols: Iterable[str],
    label: str,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Filtro agnóstico que remove linhas com valores categóricos claramente indefinidos.

    Para cada coluna candidata existente no dataframe, remove linhas cujo valor
    normalizado esteja em ``UNDEFINED_CATEGORICAL_TOKENS`` (Tx, NX, Unknown, etc.).
    É seguro aplicar em qualquer coluna porque só atinge tokens que são
    universalmente reconhecidos como "valor não classificado".

    Isso resolve o caso em que o usuário não precisa configurar filtros
    manualmente, mas ainda quer descartar samples claramente incompletos —
    análogo ao filtro ``sampleDF.T_STAGE != "Tx"`` do notebook de referência,
    sem depender do nome específico da coluna.
    """
    out = df
    initial_size = len(out)
    for col in candidate_cols or []:
        if col not in out.columns:
            continue
        normalized = out[col].apply(_normalize_token)
        before = len(out)
        out = out[~normalized.isin(UNDEFINED_CATEGORICAL_TOKENS)].copy()
        if len(out) < before:
            _log(f"{label}: drop_undefined removed {before - len(out)} row(s) on {col}")
    total_removed = initial_size - len(out)
    if total_removed > 0:
        _emit(
            progress_callback,
            f"{label}: drop_undefined removeu {total_removed} linha(s) com valores indefinidos ({initial_size}→{len(out)}).",
            None,
        )
    return out


def _apply_outliers_iqr(df: pd.DataFrame, factors: Dict[str, float], label: str) -> pd.DataFrame:
    out = df
    for col, factor in (factors or {}).items():
        if col not in out.columns:
            continue
        thr = _iqr_upper_threshold(out[col], float(factor))
        if thr is None:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        before = len(out)
        out = out[numeric < thr].copy()
        _log(f"{label}: removed outliers for {col} ({before} -> {len(out)})")
    return out


def _apply_discretization(df: pd.DataFrame, cols: List[str], bins: int, label: str) -> None:
    # IMPORTANTE: a versão discretizada substitui a coluna original (mesmo nome).
    # Isso mantém o CombinedDF com nomes limpos (sem sufixos como __discrete)
    # e replica a semântica do notebook de referência (que faz rename
    # Age_Discrete -> Age, TMB_Discrete -> TMB, SPY_Discrete -> SPY).
    for col in cols or []:
        if col in df.columns:
            df[col] = discretize_in_groups(df[col], bins)
            _log(f"{label}: discretized {col}")


POSITIVE_OUTCOME_TOKENS = {
    "deceased",
    "dead",
    "died",
    "death",
    "expired",
    "yes",
    "true",
    "positive",
    "event",
    "progressed",
    "progression",
    "recurred",
    "recurrence",
    "relapsed",
    "relapse",
    "1",
    "1.0",
    "1:deceased",
}

NEGATIVE_OUTCOME_TOKENS = {
    "living",
    "alive",
    "no",
    "false",
    "negative",
    "censored",
    "diseasefree",
    "disease_free",
    "disease free",
    "no event",
    "non progressed",
    "non-progressed",
    "no progression",
    "0",
    "0.0",
    "0:living",
}


def _normalize_outcome_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    return text


def _infer_binary_outcome_value(value: object) -> object:
    text = _normalize_outcome_value(value)

    if not text:
        return pd.NA

    compact = text.replace(" ", "_")

    if text in POSITIVE_OUTCOME_TOKENS or compact in POSITIVE_OUTCOME_TOKENS:
        return 1

    if text in NEGATIVE_OUTCOME_TOKENS or compact in NEGATIVE_OUTCOME_TOKENS:
        return 0

    if any(token in text for token in ["deceased", "dead", "died", "death", "expired"]):
        return 1

    if any(token in text for token in ["living", "alive"]):
        return 0

    if any(token in text for token in ["progression", "progressed", "recurred", "recurrence", "relapse", "relapsed"]):
        return 1

    if any(token in text for token in ["no progression", "no event", "disease free", "censored"]):
        return 0

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if float(numeric) == 1.0:
            return 1
        if float(numeric) == 0.0:
            return 0

    return pd.NA


def _apply_outcome_binary(
    df: pd.DataFrame,
    cols: List[str],
    positive_regex: str,
    label: str,
    progress_callback: ProgressCallback = None,
    min_coverage: float = 0.70,
) -> None:
    use_auto = str(positive_regex).strip().lower() in {"", "__auto__", "auto"}

    # IMPORTANTE: a versão binária substitui a coluna original (mesmo nome).
    # Isso mantém o CombinedDF com nomes limpos (sem sufixos como __binary)
    # e replica a semântica do notebook de referência (que mantém o nome
    # OS_STATUS / Status mesmo após a conversão para 0/1).
    for col in cols or []:
        if col not in df.columns:
            continue

        if use_auto:
            values = df[col].apply(_infer_binary_outcome_value).astype("Int64")
            mapped = int(values.notna().sum())
            total = int(len(values))
            coverage = (mapped / total) if total > 0 else 0.0

            df[col] = values

            _log(f"{label}: binary outcome {col} ({mapped}/{total} mapped, {coverage:.0%})")

            # Aviso só quando cobertura baixa de fato — vai pra UI por ser
            # informação acionável para o usuário (pode ter escolhido a
            # coluna errada).
            if total > 0 and coverage < min_coverage:
                _emit(
                    progress_callback,
                    f"AVISO | outcome '{col}' mapeado em apenas {coverage:.0%} ({mapped}/{total}).",
                    None,
                )
        else:
            rx = re.compile(str(positive_regex), flags=re.IGNORECASE)
            s = df[col].astype("string")
            df[col] = s.apply(
                lambda v: 1 if (v is not pd.NA and rx.search(str(v) or "") is not None) else 0
            ).astype("Int64")

            _log(f"{label}: binary outcome {col} via regex")

def _apply_stage_roman(df: pd.DataFrame, cols: List[str], roman_regex: str, label: str) -> None:
    rx = re.compile(str(roman_regex), flags=re.IGNORECASE)

    def _conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        m = rx.search(str(v))
        return _roman_to_int(m.group(1)) if m else pd.NA

    # IMPORTANTE: a versão numérica substitui a coluna original (mesmo nome).
    # Mantém o CombinedDF limpo (sem sufixos __num) e alinhado ao notebook
    # de referência.
    for col in cols or []:
        if col in df.columns:
            df[col] = df[col].apply(_conv).astype("Int64")
            _log(f"{label}: roman stage {col}")


def _parse_gmt(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathway_name = str(parts[0]).strip()
                genes = _normalize_gene_collection(parts[2:])
                if pathway_name and genes:
                    out[pathway_name] = genes
    return out


def read_driver_genes(cfg: PreprocessingConfig) -> set[str]:
    if cfg.drivers_file and cfg.drivers_file.exists():
        df = pd.read_csv(cfg.drivers_file, sep="\t", low_memory=False)
        gene_col = "symbol" if "symbol" in df.columns else df.columns[0]
        drivers = set(_normalize_gene_collection(df[gene_col].dropna().astype(str).tolist()))
        _log(f"Loaded drivers={len(drivers)} from {cfg.drivers_file}")
        return drivers
    _log("No drivers file provided. Using all genes from MAF as eligible drivers.")
    return set()


def load_pathway_hierarchy(cfg: PreprocessingConfig) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    geneset_pkl = cfg.pathway_geneset_pickle
    networks_pkl = cfg.pathway_networks_pickle
    if geneset_pkl and networks_pkl and geneset_pkl.exists() and networks_pkl.exists():
        with open(geneset_pkl, "rb") as f:
            pathway_gene_set = pickle.load(f)
        with open(networks_pkl, "rb") as f:
            pathway_networks = pickle.load(f)
        pathway_gene_set = {
            str(name).strip(): _normalize_gene_collection(genes)
            for name, genes in dict(pathway_gene_set).items()
            if str(name).strip()
        }
        pathway_networks = {str(name).strip(): graph for name, graph in dict(pathway_networks).items() if str(name).strip()}
        return pathway_gene_set, pathway_networks
    return {}, {}


def significant_intersection(neither: int, anotb: int, bnota: int, both: int) -> Dict[str, float]:
    oddsratio, p_fisher = stats.fisher_exact([[both, anotb], [bnota, neither]], alternative="two-sided")
    return {
        "OR": float(oddsratio),
        "pValue_raw": float(p_fisher),
        "pValue": round(float(p_fisher), 3),
    }


def _bfs_nodes(graph: Any, start: str) -> List[str]:
    visited: Set[str] = {start}
    out: List[str] = []
    q = deque([start])
    while q:
        u = q.popleft()
        out.append(u)
        try:
            neigh = list(graph.neighbors(u))
        except Exception:
            neigh = []
        for v in neigh:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return out


def _deduplicate_by_first(
    df: pd.DataFrame,
    key_col: str,
    label: str = "",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    if key_col not in df.columns:
        return df
    before = len(df)
    out = df.dropna(subset=[key_col]).drop_duplicates(subset=[key_col], keep="first").copy()
    removed = before - len(out)
    if removed > 0:
        # Apenas stdout — não polui o histórico da UI.
        _log(f"{label or 'dataset'}: removed {removed} duplicate(s)/null(s) on {key_col} ({before} -> {len(out)})")
    return out


def _series_to_clean_dict(row: pd.Series, cols: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        out[c] = _clean_scalar(row[c]) if c in row.index else pd.NA
    return out

def _build_pathway_catalog(
    pathway_gene_set: Dict[str, List[str]],
    pathway_networks: Dict[str, Any],
    include_subpathways: bool,
    selected_super_pathways: Optional[List[str]],
) -> List[Dict[str, Any]]:

    catalog: List[Dict[str, Any]] = []
    seen_output_names: Set[str] = set()

    if selected_super_pathways:
        roots = [str(root).strip() for root in selected_super_pathways if str(root).strip()]
    elif pathway_networks:
        roots = sorted(str(root).strip() for root in pathway_networks.keys() if str(root).strip())
    else:
        roots = sorted(str(pathway).strip() for pathway in pathway_gene_set.keys() if str(pathway).strip())

    if pathway_networks:
        for root in roots:
            if root not in pathway_networks:
                continue

            nodes = _bfs_nodes(pathway_networks[root], root)

            for node in nodes:
                node = str(node).strip()

                if not node:
                    continue

                if node == root:
                    output_name = root
                    level = "super"
                else:
                    if not include_subpathways:
                        continue
                    output_name = f"{root} -- {node}"
                    level = "sub"

                if output_name in seen_output_names:
                    continue

                genes = pathway_gene_set.get(node)

                if not genes:
                    continue

                seen_output_names.add(output_name)

                catalog.append(
                    {
                        "pathway": node,
                        "root": root,
                        "level": level,
                        "output_name": output_name,
                        "genes": genes,
                    }
                )

    else:
        for pathway_name, genes in pathway_gene_set.items():
            pathway_name = str(pathway_name).strip()

            if not pathway_name or pathway_name in seen_output_names:
                continue

            seen_output_names.add(pathway_name)

            catalog.append(
                {
                    "pathway": pathway_name,
                    "root": pathway_name,
                    "level": "standalone",
                    "output_name": pathway_name,
                    "genes": genes,
                }
            )

    catalog.sort(
        key=lambda x: (
            str(x["root"]),
            0 if x["level"] == "super" else 1,
            str(x["pathway"]),
            str(x["output_name"]),
        )
    )

    return catalog

def prepare_data(cfg: PreprocessingConfig, progress_callback: ProgressCallback = None) -> dict[str, Any]:
    if not cfg.selection:
        raise ValueError("PreprocessingConfig.selection is required.")

    selection = cfg.selection
    datasets_dir = cfg.paths.tables_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, "Validando configuração e preparando diretórios de saída.", 0.02)
    reference_files = _apply_default_reference_files(cfg)
    # Caminhos completos vão só para stdout (informativos mas longos).
    _log(
        " | ".join(
            f"{key}: {reference_files[key] or 'não encontrado'}"
            for key in ("drivers", "geneset", "networks", "gmt")
        )
    )
    missing = [k for k, v in reference_files.items() if not v]
    if missing:
        _emit(
            progress_callback,
            f"Arquivos de referência não encontrados em data/: {', '.join(missing)}.",
            0.10,
        )
    else:
        _emit(progress_callback, "Arquivos de referência carregados.", 0.10)

    _emit(progress_callback, "Lendo arquivo MAF (detecção automática de skiprows).", 0.12)
    # Detecção robusta: conta linhas iniciais começando com '#'. Isso funciona
    # para o cBioPortal (METABRIC abre com '#Sequenced_Samples: ...'), TCGA
    # (linhas '#version 2.4', '#hugo_symbol gene_name', etc.) e quaisquer
    # outros MAFs que sigam essa convenção. Cai num fallback de candidatos
    # apenas se a detecção automática não funcionar.
    auto_skip = _detect_skiprows_by_comment(cfg.mutations_file, comment_char="#")
    skiprows_candidates = [auto_skip] + [
        int(sk) for sk in (cfg.maf_skiprows_candidates or []) if int(sk) != auto_skip
    ]
    maf = None
    used_skip: Optional[int] = None
    for sk in skiprows_candidates:
        try:
            candidate = read_table(cfg.mutations_file, usecols=selection.maf_columns, skiprows=int(sk))
            if candidate.shape[1] > 1:
                maf = candidate
                used_skip = int(sk)
                break
        except Exception:
            continue
    if maf is None:
        raise ValueError("Could not read MAF file.")
    _log(f"MAF loaded with skiprows={used_skip}: {len(maf)} rows, {maf.shape[1]} cols")

    _emit(progress_callback, "Lendo tabelas clínicas de sample e patient.", 0.18)
    sample = read_table(cfg.clinical_sample_file, usecols=selection.sample_columns)
    patient = read_table(cfg.clinical_patient_file, usecols=selection.patient_columns)
    maf = normalize_columns(maf)
    sample = normalize_columns(sample)
    patient = normalize_columns(patient)

    ensure_required_columns(maf, [selection.maf_gene_col, selection.maf_sample_col], "MAF")
    ensure_required_columns(sample, [selection.patient_id_col, selection.sample_id_col], "SAMPLE")
    ensure_required_columns(patient, [selection.patient_id_col], "PATIENT")

    _emit(progress_callback, "Normalizando colunas, IDs e símbolos gênicos.", 0.25)
    maf[selection.maf_gene_col] = maf[selection.maf_gene_col].map(_normalize_gene_symbol)
    maf[selection.maf_sample_col] = maf[selection.maf_sample_col].map(_clean_scalar).astype("string")
    sample[selection.patient_id_col] = sample[selection.patient_id_col].map(_clean_scalar).astype("string")
    sample[selection.sample_id_col] = sample[selection.sample_id_col].map(_clean_scalar).astype("string")
    patient[selection.patient_id_col] = patient[selection.patient_id_col].map(_clean_scalar).astype("string")

    maf = maf[(maf[selection.maf_gene_col] != "") & (maf[selection.maf_sample_col].notna())].copy()
    sample = sample[sample[selection.patient_id_col].notna() & sample[selection.sample_id_col].notna()].copy()
    patient = patient[patient[selection.patient_id_col].notna()].copy()

    # ======================================================================
    # PROBLEMA 1 — background_genes a partir do MAF cru, antes de qualquer
    # restrição por samples clínicos válidos. Equivale a:
    #
    #     backgroundGenes = set(MAF.Hugo_Symbol)   # notebook célula 51
    #
    # Capturamos esse universo AGORA, antes de aplicar filtros adicionais.
    # Esse conjunto entra no teste de Fisher de cada paciente como o universo
    # total de genes "background". Se calcularmos depois (sobre `maf`
    # restrito a samples clínicos válidos), o teste de Fisher fica enviesado
    # e diverge do notebook.
    # ======================================================================
    background_genes: Set[str] = set(
        maf[selection.maf_gene_col]
        .dropna()
        .astype(str)
        .loc[lambda s: s != ""]
        .unique()
    )
    _log(f"background_genes = {len(background_genes)} (do MAF cru)")

    _emit(progress_callback, "Aplicando transformações configuradas no sample e no patient.", 0.32)
    _apply_categorical_remap(sample, cfg.categorical_remap, "SAMPLE")
    _apply_categorical_remap(patient, cfg.categorical_remap, "PATIENT")

    # PROBLEMA 6 — Filtros de linha agnósticos:
    # (a) row_filters explícitos do usuário no formato {coluna: [valores]}.
    # (b) Heurística segura que remove valores universalmente reconhecidos
    #     como "indefinido" (Tx, NX, Unknown, N/A, ...) em qualquer coluna
    #     selecionada pelo usuário. Diferente do notebook (que codificava
    #     `sampleDF.T_STAGE != "Tx"` hardcoded), aqui o filtro funciona
    #     mesmo quando a coluna se chama TUMOR_STAGE, T_Stage ou nada disso.
    sample = _apply_row_filters(sample, getattr(cfg, "row_filters_sample", None) or {}, "SAMPLE", progress_callback)
    patient = _apply_row_filters(patient, getattr(cfg, "row_filters_patient", None) or {}, "PATIENT", progress_callback)
    if getattr(cfg, "drop_undefined_categorical", True):
        sample = _apply_drop_undefined(sample, selection.sample_columns, "SAMPLE", progress_callback)
        patient = _apply_drop_undefined(patient, selection.patient_columns, "PATIENT", progress_callback)

    sample = _apply_outliers_iqr(sample, cfg.outlier_iqr_factors_sample, "SAMPLE")
    patient = _apply_outliers_iqr(patient, cfg.outlier_iqr_factors_patient, "PATIENT")
    _apply_discretization(sample, cfg.discretize_cols_sample, cfg.discretize_bins, "SAMPLE")
    _apply_discretization(patient, cfg.discretize_cols_patient, cfg.discretize_bins, "PATIENT")
    if cfg.convert_outcome_to_binary:
        _apply_outcome_binary(
            sample, cfg.outcome_binary_columns_sample, cfg.outcome_positive_regex,
            "SAMPLE", progress_callback=progress_callback,
        )
        _apply_outcome_binary(
            patient, cfg.outcome_binary_columns_patient, cfg.outcome_positive_regex,
            "PATIENT", progress_callback=progress_callback,
        )
    if cfg.convert_stage_to_numeric:
        _apply_stage_roman(sample, cfg.stage_roman_columns_sample, cfg.stage_roman_regex, "SAMPLE")
        _apply_stage_roman(patient, cfg.stage_roman_columns_patient, cfg.stage_roman_regex, "PATIENT")

    _emit(progress_callback, "Consolidando pacientes e amostras válidas e removendo duplicatas.", 0.40)
    sample = _deduplicate_by_first(sample, selection.sample_id_col, "SAMPLE", progress_callback)
    patient = _deduplicate_by_first(patient, selection.patient_id_col, "PATIENT", progress_callback)

    valid_patients = set(patient[selection.patient_id_col].dropna().astype(str))
    sample = sample[sample[selection.patient_id_col].isin(valid_patients)].copy()
    valid_samples = set(sample[selection.sample_id_col].dropna().astype(str))
    maf = maf[maf[selection.maf_sample_col].isin(valid_samples)].copy()

    drivers = read_driver_genes(cfg)

    if drivers:
        maf_filtered = maf[maf[selection.maf_gene_col].isin(drivers)].copy()
    else:
        maf_filtered = maf.copy()

    maf_filtered = (
        maf_filtered[[selection.maf_sample_col, selection.maf_gene_col]]
        .dropna()
        .copy()
    )

    driver_genes_in_dataset: Set[str] = set(
        maf_filtered[selection.maf_gene_col]
        .dropna()
        .astype(str)
        .map(_normalize_gene_symbol)
        .loc[lambda s: s != ""]
        .unique()
    )

    _log(f"background={len(background_genes)} | driver_genes_in_dataset={len(driver_genes_in_dataset)}")
    if progress_callback is not None:
        progress_callback("", 0.48)

    _emit(progress_callback, "Carregando pathways do Reactome.", 0.54)
    pathway_gene_set, pathway_networks = load_pathway_hierarchy(cfg)
    gmt_gene_sets: Dict[str, List[str]] = {}
    if cfg.use_pathways and cfg.pathway_gmt_file and cfg.pathway_gmt_file.exists():
        gmt_gene_sets = _parse_gmt(cfg.pathway_gmt_file)

    if not pathway_gene_set:
        pathway_gene_set = gmt_gene_sets
    elif gmt_gene_sets:
        for pathway_name, genes in gmt_gene_sets.items():
            pathway_gene_set.setdefault(pathway_name, genes)

    if not cfg.use_pathways or not pathway_gene_set:
        raise ValueError("No pathway definitions were found in data/.")

    # Normalização defensiva: garante que os genes vindos de pickle e/ou GMT
    # estejam no mesmo formato dos genes do MAF e da lista de drivers.
    pathway_gene_set = {
        str(pathway_name).strip(): _normalize_gene_collection(genes)
        for pathway_name, genes in pathway_gene_set.items()
        if str(pathway_name).strip()
    }

    all_pathway_genes: Set[str] = set().union(*(set(genes) for genes in pathway_gene_set.values()))
    maf_clinical_genes: Set[str] = set(
        maf[selection.maf_gene_col]
        .dropna()
        .astype(str)
        .map(_normalize_gene_symbol)
        .loc[lambda s: s != ""]
        .unique()
    )

    maf_pathway_overlap = maf_clinical_genes & all_pathway_genes
    background_pathway_overlap = background_genes & all_pathway_genes
    driver_pathway_overlap = driver_genes_in_dataset & all_pathway_genes

    # Diagnóstico de genes vai só para stdout do servidor (não polui a UI).
    _log(
        f"GENES | maf_clinical={len(maf_clinical_genes)} "
        f"all_pathway={len(all_pathway_genes)} "
        f"maf∩pathways={len(maf_pathway_overlap)} "
        f"bg∩pathways={len(background_pathway_overlap)} "
        f"driver∩pathways={len(driver_pathway_overlap)}"
    )
    # Atualiza só a barra de progresso, sem mensagem na lista.
    if progress_callback is not None:
        progress_callback("", 0.56)

    if drivers and (maf_filtered.empty or (not driver_pathway_overlap and maf_pathway_overlap)):
        _emit(
            progress_callback,
            "AVISO | Filtro de drivers eliminou genes; usando MAF clínico completo (fallback).",
            0.57,
        )
        maf_filtered = (
            maf[[selection.maf_sample_col, selection.maf_gene_col]]
            .dropna()
            .copy()
        )
        driver_genes_in_dataset = set(
            maf_filtered[selection.maf_gene_col]
            .dropna()
            .astype(str)
            .map(_normalize_gene_symbol)
            .loc[lambda s: s != ""]
            .unique()
        )

    # Diferente da versão anterior do service, não montamos o catálogo inteiro via BFS.
    # O notebook faz uma busca incremental: começa pelos superpathways e só aprofunda
    # nos filhos quando o pathway atual não atinge o número mínimo de pacientes.
    initial_pathways = [
        str(name).strip()
        for name in (cfg.super_pathways_names or DEFAULT_SUPER_PATHWAYS)
        if str(name).strip()
    ]

    number_of_patients = int(maf_filtered[selection.maf_sample_col].nunique())
    min_patients = max(1, int(round(number_of_patients * cfg.subgroup_min_fraction)))

    # Estratégia de preenchimento dos valores de pathway. Lê do cfg com
    # fallback seguro para o comportamento atual ("fisher_masked"). Ver
    # constante PATHWAY_VALUE_STRATEGIES no topo do arquivo para detalhes.
    pathway_value_strategy = str(
        getattr(cfg, "pathway_value_strategy", "fisher_masked") or "fisher_masked"
    ).strip().lower()
    if pathway_value_strategy not in PATHWAY_VALUE_STRATEGIES:
        _emit(
            progress_callback,
            f"AVISO | pathway_value_strategy='{pathway_value_strategy}' inválido. "
            f"Usando 'fisher_masked'. Opções válidas: {PATHWAY_VALUE_STRATEGIES}.",
            None,
        )
        pathway_value_strategy = "fisher_masked"

    # Threshold de cobertura para coverage_dense. Default 0.80 (80%) — mesmo
    # valor do PreProcessEnhanced.
    coverage_min = float(getattr(cfg, "coverage_min_fraction", 0.80) or 0.80)

    _log(f"PATHWAY_STRATEGY={pathway_value_strategy} coverage_min={coverage_min}")

    # Stats detalhadas só no stdout — informativas mas verbosas demais para a UI.
    _log(
        f"STATS | bg_genes={len(background_genes)} "
        f"maf_rows={len(maf)} filtered_rows={len(maf_filtered)} "
        f"samples_with_mut={maf_filtered[selection.maf_sample_col].nunique()} "
        f"driver_genes={len(driver_genes_in_dataset)} "
        f"pathway_set={len(pathway_gene_set)} pathway_nets={len(pathway_networks)} "
        f"min_patients={min_patients} initial_super_pathways={len(initial_pathways)}"
    )
    _emit(
        progress_callback,
        f"Iniciando busca: {number_of_patients} samples, mínimo por pathway = {min_patients}.",
        0.61,
    )

    # PROBLEMA 4 — Após as transformações (_apply_discretization,
    # _apply_outcome_binary, _apply_stage_roman), a coluna transformada
    # SUBSTITUI a coluna original com o mesmo nome. Por isso não há mais
    # sufixos __discrete/__binary/__num para acumular aqui — basta usar
    # diretamente as colunas selecionadas pelo usuário, exceto os IDs.
    out_sample_cols = [
        c for c in selection.sample_columns
        if c in sample.columns
        and c not in {selection.patient_id_col, selection.sample_id_col}
    ]
    out_patient_cols = [
        c for c in selection.patient_columns
        if c in patient.columns and c != selection.patient_id_col
    ]

    sample_by_id = sample.set_index(selection.sample_id_col, drop=False)
    patient_by_id = patient.set_index(selection.patient_id_col, drop=False)

    sample_gene_map: Dict[str, Set[str]] = {
        str(sample_id): {
            _normalize_gene_symbol(g)
            for g in df_grp[selection.maf_gene_col].astype(str).tolist()
            if _normalize_gene_symbol(g)
        }
        for sample_id, df_grp in maf_filtered.groupby(selection.maf_sample_col)
    }

    group_resume_records: List[Dict[str, Any]] = []
    patient_vs_pathways: Dict[str, Dict[str, int]] = {}
    datasets_written: List[Path] = []
    datasets: Dict[str, pd.DataFrame] = {}

    # PROBLEMA 4 — Os datasets individuais usam os nomes "limpos" das colunas
    # clínicas. Mantemos o prefixo SAMPLE__/PATIENT__ APENAS para colunas que
    # têm o mesmo nome nos dois arquivos (colisão), de modo a preservar
    # ambos os valores. Caso contrário, o nome original é usado, igual
    # ao notebook de referência.
    sample_patient_overlap = set(out_sample_cols) & set(out_patient_cols)

    def _out_sample_col_name(c: str) -> str:
        return f"SAMPLE__{c}" if c in sample_patient_overlap else c

    def _out_patient_col_name(c: str) -> str:
        return f"PATIENT__{c}" if c in sample_patient_overlap else c

    def _evaluate_pathway(pathway_info: Dict[str, Any]) -> bool:
        output_name = str(pathway_info["output_name"])
        pathway_name = str(pathway_info["pathway"])
        root_name = str(pathway_info["root"])
        level = str(pathway_info["level"])

        # Não devemos reduzir o pathway para genes presentes no MAF filtrado.
        # - tamanho do pathway
        # - BnotA
        # - interseção com os genes mutados do paciente
        genes_pathway: Set[str] = set(_normalize_gene_collection(pathway_info.get("genes", [])))

        if not genes_pathway:
            return False

        patient_vs_pathways[output_name] = {}

        df_out: Dict[str, List[object]] = {
            "PatientID": [],
            "SampleID": [],
            "PathwayRoot": [],
            "PathwayName": [],
            "PathwayLevel": [],
        }

        for c in out_sample_cols:
            df_out[_out_sample_col_name(c)] = []

        for c in out_patient_cols:
            # Evita sobrescrever uma chave de SAMPLE__ já criada quando o nome
            # da coluna não está em colisão mas casualmente coincide com algo.
            key = _out_patient_col_name(c)
            if key not in df_out:
                df_out[key] = []

        df_out[output_name] = []

        max_intersection = 0
        raw_patient_hits: Set[str] = set()
        max_raw_intersection = 0
        best_pvalue: Optional[float] = None
        best_or: Optional[float] = None

        # PASSO 1 — Loop de coleta. Iteramos por TODOS os samples e calculamos
        # `both` e (quando relevante) o teste de Fisher. Acumulamos os
        # resultados em buffers; só decidimos depois o que vai para o df_out
        # com base na estratégia escolhida.
        #
        # Estrutura: lista de tuplas
        #   (sample_id, patient_id, srow, prow, both, fisher_passed)
        # onde `fisher_passed` é True quando OR>thr E p<thr (irrelevante para
        # coverage_dense, mas calculamos sempre para deixar o log consistente).
        candidates: List[Tuple[str, str, pd.Series, pd.Series, int, bool]] = []
        # Contagem de pacientes únicos com both>0 (usada por coverage_dense).
        patients_with_intersection: Set[str] = set()

        for sample_id, genes_samples in sample_gene_map.items():
            if sample_id not in sample_by_id.index:
                continue

            srow = sample_by_id.loc[sample_id]
            patient_id = str(srow[selection.patient_id_col])

            if patient_id not in patient_by_id.index:
                continue

            prow = patient_by_id.loc[patient_id]

            # background_genes = set(MAF.Hugo_Symbol)
            # genes_samples = drivers mutados naquela amostra
            # genes_pathway = genes completos do pathway
            neither = len(background_genes - genes_samples - genes_pathway)
            anotb = len(genes_samples - genes_pathway)
            bnota = len(genes_pathway - genes_samples)
            both = len(genes_samples & genes_pathway)

            fisher_passed = False

            if both > 0:
                raw_patient_hits.add(patient_id)
                max_raw_intersection = max(max_raw_intersection, int(both))
                patients_with_intersection.add(patient_id)

                # Calculamos Fisher só quando `both>0` (caso both=0 com Fisher
                # nunca dá enriquecimento positivo). Isso preserva a mesma
                # otimização do código anterior.
                result = significant_intersection(
                    neither=neither,
                    anotb=anotb,
                    bnota=bnota,
                    both=both,
                )

                if best_pvalue is None or result["pValue_raw"] < best_pvalue:
                    best_pvalue = result["pValue_raw"]
                if np.isfinite(result["OR"]):
                    best_or = result["OR"] if best_or is None else max(best_or, result["OR"])

                fisher_passed = (
                    result["pValue"] < cfg.fisher_pvalue_threshold
                    and result["OR"] > cfg.fisher_or_threshold
                )

            candidates.append((sample_id, patient_id, srow, prow, int(both), fisher_passed))

        # PASSO 2 — Decidir se o pathway será salvo, segundo a estratégia.
        # As três estratégias usam o MESMO universo de samples avaliados acima;
        # o que muda é o critério de seleção.
        total_samples_evaluated = len(candidates)

        if pathway_value_strategy == "coverage_dense":
            # Cobertura: fração de pacientes com pelo menos 1 driver mutado
            # no pathway. Sem teste de Fisher. Threshold = coverage_min.
            unique_patients_all = len({p for _, p, _, _, _, _ in candidates})
            patients_hit_count = len(patients_with_intersection)
            coverage = (
                patients_hit_count / unique_patients_all
                if unique_patients_all > 0
                else 0.0
            )
            is_saving = coverage >= coverage_min

            # Usados para o log abaixo (com nomes paralelos aos das outras estratégias).
            sample_hits_count = sum(1 for _, _, _, _, b, _ in candidates if b > 0)
            patient_count = patients_hit_count
        else:
            # fisher_masked OU fisher_dense: ambos usam o mesmo critério de
            # seleção (Fisher OR>thr AND p<thr, threshold = min_patients).
            # PROBLEMA 3 — contagem por sample salvo (igual ao notebook).
            sample_hits_count = sum(
                1 for _, _, _, _, b, ok in candidates if ok and b > 0
            )
            is_saving = sample_hits_count >= min_patients
            patient_count = sample_hits_count
            coverage = None

        # PASSO 3 — Quando o pathway é salvo, popula o df_out de acordo com
        # a estratégia: máscara (apenas pacientes que passaram Fisher) ou
        # denso (todos os pacientes elegíveis, valor=both, 0 quando sem
        # mutação no pathway).
        if is_saving:
            for sample_id, patient_id, srow, prow, both, fisher_passed in candidates:
                if pathway_value_strategy == "fisher_masked":
                    # Comportamento original: só inclui se passou Fisher.
                    if not fisher_passed:
                        continue
                    value = both
                else:
                    # fisher_dense ou coverage_dense: inclui SEMPRE, com
                    # value=both. Para pacientes sem mutação no pathway,
                    # both é 0 e fica 0 mesmo (sem máscara MAR/MNAR).
                    value = both

                # Mantém o maior valor por paciente (se houver múltiplas
                # amostras do mesmo paciente). Para 0, não atualiza para
                # garantir que um sample com `both>0` posterior sobrescreva.
                previous_value = patient_vs_pathways[output_name].get(patient_id)
                if previous_value is None or value > previous_value:
                    patient_vs_pathways[output_name][patient_id] = value

                if value > max_intersection:
                    max_intersection = value

                df_out["PatientID"].append(patient_id)
                df_out["SampleID"].append(str(sample_id))
                df_out["PathwayRoot"].append(root_name)
                df_out["PathwayName"].append(pathway_name)
                df_out["PathwayLevel"].append(level)

                sample_values = _series_to_clean_dict(srow, out_sample_cols)
                patient_values = _series_to_clean_dict(prow, out_patient_cols)

                for c in out_sample_cols:
                    df_out[_out_sample_col_name(c)].append(sample_values[c])

                for c in out_patient_cols:
                    df_out[_out_patient_col_name(c)].append(patient_values[c])

                df_out[output_name].append(value)

        # PROBLEMA 3 — log estilo notebook. Para coverage_dense, mostramos
        # também o % de cobertura.
        unique_patients_count = len(patient_vs_pathways[output_name])

        coverage_str = (
            f" coverage={coverage:.1%}" if coverage is not None else ""
        )

        # Stdout: log detalhado para auditoria pós-execução (não polui a UI).
        _log(
            f"{'SAVING ' if is_saving else 'DEEPING'} {output_name} | "
            f"samples={patient_count} unique_pat={unique_patients_count} "
            f"raw={len(raw_patient_hits)} maxInt={max_intersection}"
            f"{coverage_str} "
            f"bestP={best_pvalue if best_pvalue is not None else 'NA'} "
            f"bestOR={best_or if best_or is not None else 'NA'}"
        )
        # UI: emite SOMENTE quando o pathway é salvo — caso raro e relevante.
        # Pathways com DEEPING são milhares; emitir todos estoura memória.
        if is_saving:
            _emit(
                progress_callback,
                f"==SAVING== {output_name} | samples={patient_count} | unique_patients={unique_patients_count} | maxInt={max_intersection}{coverage_str}",
                None,
            )

        if is_saving:
            df_dataset = pd.DataFrame(df_out)
            out_path = datasets_dir / f"{safe_filename(output_name)}.tsv"
            df_dataset.to_csv(out_path, sep="\t", index=False)

            datasets_written.append(out_path)
            datasets[output_name] = df_dataset

            group_resume_records.append(
                {
                    "Pathway": output_name,
                    "PathwayRoot": root_name,
                    "PathwayName": pathway_name,
                    "PathwayLevel": level,
                    "SP len": int(len(genes_pathway)),
                    # Patients = samples salvos com value>0 (igual ao notebook).
                    "Patients": int(patient_count),
                    "Patients %": round((patient_count / max(number_of_patients, 1)) * 100, 3),
                    "Ratio Pat/len": round(patient_count / max(len(genes_pathway), 1), 3),
                    "MaxIntersection": int(max_intersection),
                    # Coluna adicional informativa: número de pacientes únicos,
                    # útil para a análise downstream e para auditar diferenças
                    # entre datasets com múltiplos samples por paciente.
                    "UniquePatients": int(unique_patients_count),
                }
            )
            return True

        return False

    def _pathway_info_from_queue_name(queue_name: str) -> Optional[Dict[str, Any]]:
        # NOTA DE DESIGN — DIFERENÇA INTENCIONAL EM RELAÇÃO AO NOTEBOOK:
        # O notebook de referência só desce UM nível na hierarquia Reactome
        # (split por " -- " em [spName, subName] e usa subName diretamente).
        # Aqui usamos parts[-1], ou seja, o último elemento do caminho. Isso
        # permite que a busca em largura (BFS) desça arbitrariamente fundo
        # na árvore — sub-subpathways, sub-sub-subpathways, etc. — enquanto
        # houver filhos cujo nome esteja em pathway_gene_set.
        #
        # Essa decisão é deliberada porque:
        #   1. Em datasets pequenos ou específicos, muitos subpathways de
        #      primeiro nível ainda não atingem o threshold mínimo de
        #      pacientes, mas filhos mais profundos podem.
        #   2. O critério de parada (solved_roots no loop principal)
        #      garante que cada superpathway só salva um descendente, então
        #      a busca não explode em duplicatas.
        #
        # Se você quiser replicar EXATAMENTE o comportamento do notebook,
        # troque `pathway_name = parts[-1]` por `pathway_name = parts[1]
        # if len(parts) > 1 else parts[0]` e ignore queue_names com 3+ partes.
        parts = [p.strip() for p in str(queue_name).split(" -- ") if p.strip()]

        if not parts:
            return None

        root_name = parts[0]
        pathway_name = parts[-1]
        level = "super" if len(parts) == 1 else "sub"

        genes = pathway_gene_set.get(pathway_name)

        if not genes:
            return None

        return {
            "pathway": pathway_name,
            "root": root_name,
            "level": level,
            "output_name": str(queue_name),
            "genes": genes,
        }

    def _children_for_queue_name(queue_name: str) -> List[str]:
        # Ver nota em _pathway_info_from_queue_name: também usamos parts[-1]
        # como nó atual, permitindo descida profunda na BFS Reactome.
        if not cfg.analyze_subpathways:
            return []

        parts = [p.strip() for p in str(queue_name).split(" -- ") if p.strip()]

        if not parts:
            return []

        root_name = parts[0]
        current_node = parts[-1]
        graph = pathway_networks.get(root_name)

        if graph is None:
            return []

        try:
            children = list(graph.neighbors(current_node))
        except Exception:
            children = []

        out: List[str] = []

        for child in children:
            child = str(child).strip()

            if not child:
                continue

            if child not in pathway_gene_set:
                continue

            out.append(f"{root_name} -- {child}")

        return sorted(set(out))

    if not maf_pathway_overlap:
        _log(
            f"MAF examples: {_debug_gene_examples(sorted(maf_clinical_genes))} | "
            f"Pathway examples: {_debug_gene_examples(sorted(all_pathway_genes))}"
        )
        _emit(
            progress_callback,
            "ERRO | Nenhum gene do MAF aparece nos pathways do Reactome. Verifique se os símbolos são HGNC.",
            None,
        )

    # Construção da fila de avaliação. O método depende da estratégia:
    #
    # - fisher_*: busca incremental igual ao notebook. Começa pelos
    #   superpathways e só desce na hierarquia Reactome quando o pathway
    #   atual não atinge o threshold (BFS controlado por solved_roots).
    #
    # - coverage_dense: itera TODOS os pathways disponíveis em
    #   pathway_gene_set, sem hierarquia. Equivale ao loop do
    #   PreProcessEnhanced sobre as chaves do GMT. O filtro de cobertura
    #   naturalmente seleciona os pathways mais relevantes (geralmente
    #   superpathways grandes, mas qualquer subpathway com cobertura alta
    #   também aparece).
    if pathway_value_strategy == "coverage_dense":
        # Ordena para resultado determinístico e prioridade visual aos pathways
        # maiores (que tendem a aparecer primeiro como SAVING).
        pathways_queue: List[str] = sorted(pathway_gene_set.keys())
        use_bfs = False
        _log(
            f"COVERAGE_DENSE: iterando {len(pathways_queue)} pathways de "
            f"pathway_gene_set sem BFS Reactome"
        )
    else:
        pathways_queue: List[str] = list(dict.fromkeys(initial_pathways))
        use_bfs = True

    queued_pathways: Set[str] = set(pathways_queue)
    solved_roots: Set[str] = set()
    pathway_catalog: List[Dict[str, Any]] = []
    idx = 0

    # Intervalo de UI: emite uma linha textual a cada N pathways processados.
    # A barra de progresso continua avançando suavemente em toda iteração.
    # Com ~2500 pathways no Reactome, emitir 1 mensagem a cada 50 mantém o
    # histórico em ~50 linhas em vez de milhares, evitando estouro de memória.
    ui_log_interval = 50

    while idx < len(pathways_queue):
        queue_name = pathways_queue[idx]
        idx += 1

        root_name = str(queue_name).split(" -- ")[0].strip()

        # solved_roots só vale para estratégias com BFS. Em coverage_dense
        # cada pathway é independente, queremos todos avaliados.
        if use_bfs and root_name in solved_roots:
            continue

        pathway_info = _pathway_info_from_queue_name(queue_name)

        if pathway_info is None:
            # Stdout apenas — caso comum e ruidoso.
            _log(f"Iter {idx}/{len(pathways_queue)}: pathway ausente na referência: {queue_name}")
            continue

        pathway_catalog.append(pathway_info)

        level = str(pathway_info.get("level", ""))
        progress = 0.62 + (0.24 * min(idx / max(len(pathways_queue), 1), 1.0))

        # Stdout sempre (auditoria completa); UI só periodicamente.
        _log(
            f"Iter {idx}/{len(pathways_queue)} "
            f"{'sub' if level == 'sub' else 'super'}: {queue_name}"
        )
        if idx % ui_log_interval == 0 or idx == len(pathways_queue):
            _emit(
                progress_callback,
                f"Iteração {idx}/{len(pathways_queue)} — avaliando: {queue_name}",
                progress,
            )
        elif progress_callback is not None:
            # Atualiza só a barra de progresso, sem texto novo na lista.
            progress_callback("", progress)

        saved = _evaluate_pathway(pathway_info)

        # BFS só faz sentido nas estratégias com Fisher. Em coverage_dense
        # não enfileiramos filhos porque já estamos varrendo tudo de uma vez.
        if use_bfs:
            if saved:
                solved_roots.add(root_name)
                continue

            for child_name in _children_for_queue_name(queue_name):
                if child_name not in queued_pathways:
                    queued_pathways.add(child_name)
                    pathways_queue.append(child_name)

    _emit(progress_callback, "Consolidando resumo final e CombinedDF.tsv.", 0.90)
    group_resume_df = pd.DataFrame(group_resume_records)
    if group_resume_df.empty:
        group_resume_df = pd.DataFrame(
            columns=[
                "Pathway",
                "PathwayRoot",
                "PathwayName",
                "PathwayLevel",
                "SP len",
                "Patients",
                "Patients %",
                "Ratio Pat/len",
                "MaxIntersection",
                "UniquePatients",
            ]
        )
    else:
        group_resume_df = group_resume_df.sort_values(
            ["Patients", "SP len", "Pathway"],
            ascending=[False, False, True],
            ignore_index=True,
        )

    group_resume_path = cfg.paths.tables_dir / "group_resume.tsv"
    group_resume_df.to_csv(group_resume_path, sep="\t", index=False)

    combined_df = None
    combined_df_path = None
    patient_vs_pathways_path = None

    # Estratégia: SEMPRE gerar CombinedDF, mesmo quando nenhum pathway foi
    # salvo. Quando há datasets salvos, usamos os pacientes presentes neles
    # (matching com o notebook 1.5). Quando NÃO há datasets, usamos todos os
    # pacientes que sobreviveram aos filtros clínicos. Isso garante que o
    # usuário sempre receba ao menos os dados clínicos consolidados,
    # mesmo em datasets de painel direcionado (METABRIC, etc.) onde o
    # background_genes restrito impede a maior parte dos pathways de
    # atingir o threshold do teste de Fisher.

    if datasets:
        # allPatients = união dos PatientID presentes nos datasets de pathways salvos.
        all_patients: List[str] = []

        for df_dataset in datasets.values():
            if "PatientID" in df_dataset.columns:
                all_patients.extend(
                    [
                        str(pid)
                        for pid in df_dataset["PatientID"].dropna().astype(str).tolist()
                        if str(pid).strip()
                    ]
                )

        patient_ids = sorted(set(all_patients))
        pathway_cols = list(datasets.keys())
    else:
        # Sem pathways salvos: usa os pacientes clínicos que têm pelo menos
        # uma amostra com mutações no MAF filtrado (mesma população que o
        # loop de avaliação consideraria). Isso preserva a semântica de
        # "pacientes elegíveis" sem depender de nenhum pathway ter passado
        # o threshold.
        samples_with_mut = set(maf_filtered[selection.maf_sample_col].astype(str).dropna().unique())
        # patient_id -> True se tem pelo menos uma amostra com mutação
        sample_to_patient = (
            sample[[selection.sample_id_col, selection.patient_id_col]]
            .dropna()
            .astype("string")
        )
        eligible = sample_to_patient[
            sample_to_patient[selection.sample_id_col].isin(samples_with_mut)
        ]
        patient_ids = sorted({str(p) for p in eligible[selection.patient_id_col].tolist() if str(p).strip()})
        pathway_cols = []

        _emit(
            progress_callback,
            (
                f"AVISO | Nenhum pathway atingiu o threshold de {cfg.subgroup_min_fraction:.0%} "
                f"({min_patients} pacientes). Isso é típico de painéis direcionados "
                f"(MAF com poucos genes, como o METABRIC). O CombinedDF será gerado "
                f"apenas com as features clínicas dos {len(patient_ids)} pacientes elegíveis."
            ),
            None,
        )

    # Features clínicas/tumorais primeiro, pathways depois.
    # Mantemos nomes genéricos e agnósticos, mas removemos prefixos no CombinedDF
    # para ficar mais parecido com o notebook de referência.
    clinical_feature_cols: List[str] = []

    for c in out_patient_cols:
        if c not in clinical_feature_cols:
            clinical_feature_cols.append(c)

    for c in out_sample_cols:
        if c not in clinical_feature_cols:
            clinical_feature_cols.append(c)

    combined_df = pd.DataFrame(
        index=patient_ids,
        columns=clinical_feature_cols + pathway_cols,
    )

    combined_df.index.name = selection.patient_id_col

    for col in clinical_feature_cols:
        combined_df[col] = combined_df[col].astype("object")

    for col in pathway_cols:
        combined_df[col] = pd.Series(pd.NA, index=combined_df.index, dtype="Int64")

    if datasets:
        # para cada dataset salvo, percorre as linhas e preenche features + pathway.
        # PROBLEMA 4 — IMPORTANTE:
        # 1. As colunas dos datasets individuais já estão salvas com nome
        #    "limpo" (sem prefixo SAMPLE__/PATIENT__), exceto quando há colisão
        #    de nome entre as duas seleções. Usamos _out_sample_col_name /
        #    _out_patient_col_name para resolver isso de forma consistente.
        # 2. Sobrescrevemos SEMPRE o valor no CombinedDF, sem checar pd.notna,
        #    exatamente como o notebook 1.5 célula 8:
        #        combinedDF.at[pId, f] = row[f]
        #    A última escrita vence, o que mantém o comportamento determinístico
        #    quando um paciente aparece em vários datasets.
        for pathway_name, df_dataset in datasets.items():
            for _, row in df_dataset.iterrows():
                patient_id = row.get("PatientID")

                if patient_id is None or pd.isna(patient_id):
                    continue

                patient_id = str(patient_id)

                if patient_id not in combined_df.index:
                    continue

                for source_col in out_patient_cols:
                    dataset_col = _out_patient_col_name(source_col)
                    if dataset_col in row.index and source_col in combined_df.columns:
                        combined_df.at[patient_id, source_col] = row[dataset_col]

                for source_col in out_sample_cols:
                    dataset_col = _out_sample_col_name(source_col)
                    if dataset_col in row.index and source_col in combined_df.columns:
                        combined_df.at[patient_id, source_col] = row[dataset_col]

                if pathway_name in row.index and pd.notna(row[pathway_name]):
                    combined_df.at[patient_id, pathway_name] = int(row[pathway_name])
    else:
        # Fallback (datasets vazios): preenche features clínicas direto dos
        # dataframes patient/sample. Pré-indexa por PATIENT_ID para evitar
        # busca linear O(N) por paciente.
        patient_dedup = (
            patient.dropna(subset=[selection.patient_id_col])
            .drop_duplicates(subset=[selection.patient_id_col], keep="first")
        )
        patient_lookup = patient_dedup.set_index(
            pd.Index(patient_dedup[selection.patient_id_col].astype(str).tolist())
        )
        sample_dedup = (
            sample.dropna(subset=[selection.patient_id_col])
            .drop_duplicates(subset=[selection.patient_id_col], keep="first")
        )
        sample_lookup = sample_dedup.set_index(
            pd.Index(sample_dedup[selection.patient_id_col].astype(str).tolist())
        )

        for pid in patient_ids:
            if pid in patient_lookup.index:
                prow = patient_lookup.loc[pid]
                for source_col in out_patient_cols:
                    if source_col in prow.index and source_col in combined_df.columns:
                        combined_df.at[pid, source_col] = prow[source_col]
            if pid in sample_lookup.index:
                srow = sample_lookup.loc[pid]
                for source_col in out_sample_cols:
                    if source_col in srow.index and source_col in combined_df.columns:
                        combined_df.at[pid, source_col] = srow[source_col]

    # Garante dtype numérico anulável nas colunas de pathway.
    for pathway_name in pathway_cols:
        combined_df[pathway_name] = pd.to_numeric(
            combined_df[pathway_name],
            errors="coerce",
        ).astype("Int64")

    combined_df_path = datasets_dir / "CombinedDF.tsv"
    combined_df.to_csv(combined_df_path, sep="\t", index=True)

    if patient_vs_pathways:
        patient_vs_pathways_df = pd.DataFrame(patient_vs_pathways)
        patient_vs_pathways_path = cfg.paths.tables_dir / "patient_vs_pathways.tsv"
        patient_vs_pathways_df.to_csv(patient_vs_pathways_path, sep="\t", index=True)

    _emit(progress_callback, "Pré-processamento concluído. Arquivos finais gerados com sucesso.", 1.0)
    return {
        "maf_filtered": maf_filtered,
        "sample": sample,
        "patient": patient,
        "group_resume": group_resume_df,
        "group_resume_path": group_resume_path,
        "datasets_dir": datasets_dir,
        "dataset_paths": datasets_written,
        "combined_df": combined_df,
        "combined_df_path": combined_df_path,
        "patient_vs_pathways": patient_vs_pathways,
        "patient_vs_pathways_path": patient_vs_pathways_path,
        "pathway_catalog": pathway_catalog,
        "reference_files": reference_files,
    }