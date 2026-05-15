from __future__ import annotations

import gc
import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

import pandas as pd
from dowhy import CausalModel

from src.domain.models import InferenceRunConfig


# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------

# Regex que casa linhas de aresta no DOT: ``  "A" -> "B";`` (com aspas).
_DOT_EDGE_RE = re.compile(r'^\s*"(.*?)"\s*->\s*"(.*?)"\s*(\[[^\]]*\])?\s*;\s*$')

# Regex que casa linhas em formato TETRAD: ``A --> B``, ``C <-- D``, ``E --- F`` etc.
_TETRAD_EDGE_RE = re.compile(r'^(.*?)\s+(-->|<--|---|o->|<-o|o-o)\s+(.*?)$')

# Marker que identifica que um texto JÁ É um arquivo DOT (não strings TETRAD).
_DOT_HEADER_RE = re.compile(r"^\s*(strict\s+)?digraph\b", re.MULTILINE)

ProgressCallback = Optional[Callable[[str, Optional[float]], None]]


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[inference] {msg}", flush=True)


def _emit(progress_callback: ProgressCallback, msg: str, progress: Optional[float] = None) -> None:
    _log(msg)
    if progress_callback is not None:
        progress_callback(msg, progress)


# ----------------------------------------------------------------------
# I/O de dataset e grafo
# ----------------------------------------------------------------------

# Heurística para detectar coluna 1ª como ID de paciente/amostra (igual a
# discovery_service para evitar inconsistências de leitura).
_ID_COLUMN_PATTERNS = re.compile(
    r"^(patient[_ ]?id|sample[_ ]?id|id|case[_ ]?id|tcga|mb|p\d+|s\d+)",
    flags=re.IGNORECASE,
)


def _looks_like_id_column(series_head: pd.Series) -> bool:
    name = str(series_head.name or "")
    if not _ID_COLUMN_PATTERNS.match(name):
        sample = series_head.dropna().astype(str).tolist()
        if not sample:
            return False
        prefixes = {s.split("-")[0] if "-" in s else s[:3] for s in sample}
        if len(prefixes) > 2:
            return False
    return series_head.is_unique


def _read_dataset(path: Path) -> pd.DataFrame:
    """Carrega o dataset detectando heurísticamente se a 1ª coluna é índice (ID).

    Aplica a mesma detecção do discovery_service para garantir consistência.
    """
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    probe = pd.read_csv(path, sep=sep, low_memory=False, nrows=50)
    if probe.shape[1] >= 2 and _looks_like_id_column(probe.iloc[:, 0]):
        return pd.read_csv(path, sep=sep, low_memory=False, index_col=0)
    return pd.read_csv(path, sep=sep, low_memory=False)


def _is_dot_file(text: str) -> bool:
    """Detecção robusta de DOT (versão anterior usava só ``"digraph" in text``)."""
    return _DOT_HEADER_RE.search(text) is not None


def _parse_dot_edges(text: str) -> Tuple[list, list, list]:
    """Parseia um texto DOT em ``(directed, undirected, other)``.

    - ``directed``: lista de tuplas ``(a, b)`` para arestas ``A -> B`` sem
      ``dir=none``. Estas são utilizáveis pelo DoWhy.
    - ``undirected``: arestas ``A -> B [dir=none]`` (geradas a partir de
      ``A --- B`` do PC). Não podem ser usadas como direcionadas; serão
      reportadas.
    - ``other``: arestas com atributos especiais (``arrowtail=odot``,
      ``arrowhead=odot``) que vêm de PAGs do FCI. Também não utilizáveis.

    Essa separação é importante porque o DoWhy precisa de um DAG; passar
    arestas não-direcionadas ou bidirecionais ambíguas causa erros opacos
    em ``identify_effect``.
    """
    directed: list = []
    undirected: list = []
    other: list = []
    for line in text.splitlines():
        m = _DOT_EDGE_RE.match(line)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        attrs = m.group(3) or ""
        if "dir=none" in attrs:
            undirected.append((a, b))
        elif "arrowtail=odot" in attrs or "arrowhead=odot" in attrs:
            other.append((a, b))
        else:
            directed.append((a, b))
    return directed, undirected, other


def _graph_to_dot(
    path: Path,
    progress_callback: ProgressCallback = None,
) -> Tuple[Any, str, dict]:
    """Carrega um grafo (DOT ou TETRAD) e retorna ``(nx.DiGraph, dot, debug)``.

    Mudanças críticas:

    - **Retorna um ``networkx.DiGraph``** em vez de só string DOT. O DoWhy
      detecta o formato de string DOT por uma regex frágil
      (``re.match(r".*graph\\s*\\{.*\\}\\s*", graph)``) que falha em duas
      situações que produzimos:
        1. DOT com nome do grafo (``digraph G {`` em vez de ``digraph {``);
        2. DOT multi-linha (``.`` não casa ``\\n`` por default).
      Passar um ``nx.DiGraph`` direto contorna o problema na origem — é o
      formato preferido pelo DoWhy.

    - **Filtra colunas de ID** (PATIENT_ID, SAMPLE_ID) das arestas. Essas
      colunas nunca deveriam aparecer como nós do grafo causal.

    - **DOT canonizado** preservado como string para auditoria (escrito em
      ``graph_used.dot``), mas o objeto efetivo passado ao DoWhy é o DiGraph.

    Arestas não-direcionadas (``A --- B``) e de PAG (``o->``, ``o-o``) são
    descartadas com aviso, igual à versão anterior.
    """
    import networkx as nx  # import local para reduzir tempo de cold-start.

    text = path.read_text(encoding="utf-8")
    debug: dict = {}

    if _is_dot_file(text):
        directed, undirected, other = _parse_dot_edges(text)
        debug["input_format"] = "dot"
    else:
        directed = []
        undirected = []
        other = []
        for line in text.splitlines():
            m = _TETRAD_EDGE_RE.match(line.strip())
            if not m:
                continue
            a = m.group(1).strip()
            rel = m.group(2)
            b = m.group(3).strip()
            if rel == "-->":
                directed.append((a, b))
            elif rel == "<--":
                directed.append((b, a))
            elif rel == "---":
                undirected.append((a, b))
            else:
                other.append((a, b))
        debug["input_format"] = "tetrad"

    # Filtro de IDs: as colunas PATIENT_ID, SAMPLE_ID nunca devem ser nós
    # causais. Se aparecerem por engano, descartamos com aviso.
    id_like_nodes: set = set()
    def _is_id_node(name: str) -> bool:
        upper = str(name).upper().strip()
        return upper in {"PATIENT_ID", "SAMPLE_ID", "TUMOR_SAMPLE_BARCODE", "CASE_ID"}

    def _filter_edges(edges: list) -> list:
        kept: list = []
        for a, b in edges:
            if _is_id_node(a) or _is_id_node(b):
                if _is_id_node(a): id_like_nodes.add(a)
                if _is_id_node(b): id_like_nodes.add(b)
                continue
            kept.append((a, b))
        return kept

    directed = _filter_edges(directed)
    undirected = _filter_edges(undirected)
    other = _filter_edges(other)

    debug["directed_edges"] = len(directed)
    debug["undirected_dropped"] = len(undirected)
    debug["pag_dropped"] = len(other)
    debug["id_nodes_dropped"] = sorted(id_like_nodes)

    if undirected or other:
        _emit(
            progress_callback,
            (
                f"AVISO | Grafo contém {len(undirected)} aresta(s) não-direcionada(s) "
                f"e {len(other)} aresta(s) tipo PAG. Estas são descartadas porque o "
                "DoWhy requer um DAG direcionado. Considere orientar manualmente ou "
                "usar um algoritmo de descoberta que produza DAG (FGES) em vez de "
                "CPDAG (PC) ou PAG (FCI)."
            ),
            None,
        )
    if id_like_nodes:
        _emit(
            progress_callback,
            (
                f"AVISO | Nós de ID descartados do grafo causal: "
                f"{', '.join(sorted(id_like_nodes))}. Colunas como PATIENT_ID "
                "não são variáveis causalmente significativas."
            ),
            None,
        )

    # Constrói o DiGraph (formato preferido pelo DoWhy).
    graph_obj = nx.DiGraph()
    nodes_set: set = set()
    for a, b in directed:
        nodes_set.add(a)
        nodes_set.add(b)
    for node in nodes_set:
        graph_obj.add_node(node)
    for a, b in directed:
        graph_obj.add_edge(a, b)

    # DOT canonizado para auditoria.
    lines = ["digraph {"]
    for node in sorted(nodes_set):
        safe = str(node).replace('"', r'\"')
        lines.append(f'  "{safe}";')
    for a, b in directed:
        safe_a = str(a).replace('"', r'\"')
        safe_b = str(b).replace('"', r'\"')
        lines.append(f'  "{safe_a}" -> "{safe_b}";')
    lines.append("}")
    dot_text = "\n".join(lines) + "\n"

    return graph_obj, dot_text, debug


def _vars_from_graph(graph_obj: Any) -> list:
    """Extrai a lista de nós do DiGraph (substitui _vars_from_dot)."""
    return sorted(graph_obj.nodes())


def _vars_from_dot(dot: str) -> list:
    """Extrai todos os nomes de nós que aparecem em arestas do DOT."""
    out: set = set()
    for ln in dot.splitlines():
        m = _DOT_EDGE_RE.match(ln)
        if m:
            out.add(m.group(1))
            out.add(m.group(2))
    return sorted(out)


# ----------------------------------------------------------------------
# Backdoor variables: leitura robusta do estimand
# ----------------------------------------------------------------------

def _extract_backdoor_vars(estimand: Any) -> list:
    """Extrai a lista de variáveis de backdoor do estimand do DoWhy.

    Compatível com várias versões do DoWhy:
    - DoWhy >= 0.9: ``estimand.get_backdoor_variables()`` é um método que
      retorna lista.
    - DoWhy 0.8 e anteriores: ``estimand.backdoor_variables`` é dicionário
      ``{estimator_name: [vars]}`` em algumas versões, lista em outras.
    - Fallback: regex sobre ``str(estimand)``.

    A versão anterior tentava só ``isinstance(value, (list, tuple, set))``,
    o que retornava lista vazia em DoWhy recente (onde o atributo é um dict)
    e caía no fallback de regex (também frágil entre versões).
    """
    # 1. Método get_backdoor_variables() — preferido (DoWhy >= 0.9).
    if hasattr(estimand, "get_backdoor_variables") and callable(estimand.get_backdoor_variables):
        try:
            value = estimand.get_backdoor_variables()
            if isinstance(value, (list, tuple, set)):
                return [str(v) for v in value]
        except Exception:
            pass

    # 2. Atributos diretos — pode ser lista OU dicionário (varia entre versões).
    for attr in ["backdoor_variables", "backdoor_vars", "backdoor_set"]:
        if not hasattr(estimand, attr):
            continue
        value = getattr(estimand, attr)
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        if isinstance(value, dict):
            # Achata: pega todas as variáveis de qualquer entrada do dict.
            collected: list = []
            seen: set = set()
            for v_list in value.values():
                if not isinstance(v_list, (list, tuple, set)):
                    continue
                for v in v_list:
                    sv = str(v)
                    if sv not in seen:
                        seen.add(sv)
                        collected.append(sv)
            if collected:
                return collected

    # 3. Fallback final: regex no str do estimand.
    s = str(estimand)
    m = re.search(r"Backdoor\s+variables?\s*:\s*\[(.*?)\]", s, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return [p.strip().strip("'\"") for p in m.group(1).split(",") if p.strip()]


def _estimand_has_identification(estimand: Any) -> bool:
    """Detecta se o estimand do DoWhy foi de fato identificado.

    Quando ``proceed_when_unidentifiable=True`` e a identificação falha,
    o estimand existe mas seus ``estimands`` interno é vazio ou ``None``,
    e ``estimate_effect`` posterior vai falhar de forma opaca. Detectamos
    isso aqui para pular o par com motivo claro.

    Critérios (qualquer um basta):
    - Atributo ``estimands`` é dict não-vazio com pelo menos um valor não-None.
    - Backdoor variables não está vazio.
    - String do estimand não contém marcadores de "no identified estimand".
    """
    # Critério 1: estimands populado.
    estimands_attr = getattr(estimand, "estimands", None)
    if isinstance(estimands_attr, dict):
        for value in estimands_attr.values():
            if value is not None and value != {}:
                return True
        return False

    # Critério 2: backdoor vars não vazias (proxy).
    if _extract_backdoor_vars(estimand):
        return True

    # Critério 3: indicação textual de não-identificação.
    s = str(estimand).lower()
    if "no such variable(s) found" in s or "no identified estimand" in s:
        return False

    # Se nada disso bate, assume que foi identificado (otimista mas seguro:
    # erros em estimate_effect serão capturados depois).
    return True


# ----------------------------------------------------------------------
# Extração estruturada de resultados do DoWhy
# ----------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Converte para float quando possível, retorna None em falha."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # Filtra NaN/inf — JSON não serializa esses por default.
    if result != result or result == float("inf") or result == float("-inf"):
        return None
    return result


def _extract_estimate_summary(estimate: Any) -> dict:
    """Extrai atributos estruturados de ``CausalEstimate`` em um dict serializável.

    O output do DoWhy via ``str(estimate)`` é texto formatado para terminal:

        *** Causal Estimate ***
        ## Identified estimand
        Estimand expression: d/d[v0] (E[y|W0,W1,W2])
        ## Realized estimand
        b: y~v0+W0+W1+W2
        Target units: ate
        ## Estimate
        Mean value: 10.0003

    Para a UI queremos os campos estruturados separados: ``estimate_value``,
    ``realized_expression``, ``target_units``, ``control_value``, etc. Essa
    função extrai o que estiver disponível, com fallback no parse textual.
    """
    summary: dict = {}

    # Mean value (atributo principal)
    summary["estimate_value"] = _safe_float(getattr(estimate, "value", None))

    # Control/treatment values (de quanto-para-quanto o efeito foi calculado)
    summary["control_value"] = getattr(estimate, "control_value", None)
    summary["treatment_value"] = getattr(estimate, "treatment_value", None)

    # Expressão "realized": ex. "b: y~v0+W2+W3+W1+W0+W4"
    realized = getattr(estimate, "realized_estimand_expr", None)
    if realized is not None:
        summary["realized_expression"] = str(realized)

    # target_units: "ate", "att", "atc", ou função
    params = getattr(estimate, "params", None)
    if isinstance(params, dict):
        target_units = params.get("target_units")
        if target_units is not None and not callable(target_units):
            summary["target_units"] = str(target_units)

    # Intervalo de confiança e teste de significância (quando habilitados).
    try:
        ci = getattr(estimate, "get_confidence_intervals", None)
        if callable(ci):
            ci_value = ci()
            if ci_value is not None:
                # CI geralmente é [lower, upper] ou lista de tuples
                summary["confidence_intervals"] = _safe_serialize_value(ci_value)
    except Exception:
        pass

    try:
        sig = getattr(estimate, "test_stat_significance", None)
        if callable(sig):
            sig_value = sig()
            if isinstance(sig_value, dict):
                summary["significance_test"] = {
                    k: _safe_serialize_value(v) for k, v in sig_value.items()
                }
    except Exception:
        # Significance test só funciona quando test_significance=True no estimate
        # E não está disponível para todos os estimators. Silencioso.
        pass

    # str() do estimate para auditoria/debug — preservado completo.
    summary["raw_text"] = str(estimate)
    return summary


def _safe_serialize_value(value: Any) -> Any:
    """Reduz tuplas/arrays/numpy a formas JSON-serializáveis."""
    if isinstance(value, (list, tuple)):
        return [_safe_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_serialize_value(v) for k, v in value.items()}
    # numpy scalar?
    try:
        if hasattr(value, "item") and callable(value.item):
            scalar = value.item()
            return scalar
    except Exception:
        pass
    f = _safe_float(value)
    if f is not None:
        return f
    return str(value)


# Conjunto de refuters cujo resultado deve ser interpretado pelo p-value
# (refuters de "negative control": estimador deve manter ou zerar o efeito).
_PVALUE_REFUTERS = {
    "placebo_treatment_refuter",
    "random_common_cause",
    "data_subset_refuter",
    "dummy_outcome_refuter",
    "bootstrap_refuter",
}


def _extract_refutation_summary(refutation: Any, refuter_name: str) -> dict:
    """Extrai atributos estruturados de ``CausalRefutation``.

    Os refuters do DoWhy retornam objetos ``CausalRefutation`` com:
      - ``estimated_effect``: efeito original (antes da refutação).
      - ``new_effect``: efeito após a transformação refutadora.
      - ``refutation_result``: dict ``{p_value, is_statistically_significant}``
        quando aplicável.
      - ``refutation_type``: nome interno do refuter.

    Também atribuímos um veredicto ``passes`` por refuter, com semântica
    específica para cada tipo:

    - **placebo_treatment_refuter**: o novo efeito DEVE ir a zero quando o
      tratamento é substituído por placebo. Se ``p_value >= 0.05``, o refuter
      NÃO consegue rejeitar a hipótese de que o novo efeito é diferente de
      zero — o estimador passa (estável). Se ``p_value < 0.05``, FALHA.
    - **random_common_cause**: o efeito não deveria mudar muito quando
      adicionamos uma causa comum aleatória. Tradicionalmente avaliado pela
      proximidade entre ``estimated_effect`` e ``new_effect``.
    - **data_subset_refuter**: idem, mas reduzindo amostras.
    - **add_unobserved_common_cause**: sensibilidade. ``new_effect`` pode ser
      tupla representando o intervalo do efeito sob diferentes intensidades
      de confounder não-observado.
    """
    summary: dict = {
        "refuter_name": refuter_name,
        "refutation_type": str(getattr(refutation, "refutation_type", refuter_name)),
    }

    summary["estimated_effect"] = _safe_serialize_value(
        getattr(refutation, "estimated_effect", None)
    )
    summary["new_effect"] = _safe_serialize_value(
        getattr(refutation, "new_effect", None)
    )

    # refutation_result: dict {p_value, is_statistically_significant, ...}
    refutation_result = getattr(refutation, "refutation_result", None)
    if isinstance(refutation_result, dict):
        summary["p_value"] = _safe_float(refutation_result.get("p_value"))
        summary["is_statistically_significant"] = bool(
            refutation_result.get("is_statistically_significant", False)
        )
    else:
        summary["p_value"] = None
        summary["is_statistically_significant"] = None

    # Veredicto interpretado para a UI.
    summary["passes"] = _interpret_refutation_pass(summary, refuter_name)

    summary["raw_text"] = str(refutation)
    return summary


def _interpret_refutation_pass(summary: dict, refuter_name: str) -> Optional[bool]:
    """Retorna True/False/None para indicar se o estimador "passou" no refuter.

    - True: o estimador é robusto a essa perturbação (esperado).
    - False: o estimador FALHA (problema detectado).
    - None: sem critério objetivo disponível (refuter de sensibilidade).
    """
    if refuter_name == "placebo_treatment_refuter":
        # placebo: NEW_EFFECT deveria ser ZERO. p < 0.05 quer dizer
        # rejeitamos H0 (new_effect != 0) → estimador FALHA.
        # p >= 0.05 → não rejeitamos H0 → estimador passa.
        p = summary.get("p_value")
        if p is None:
            return None
        return p >= 0.05

    if refuter_name in {"random_common_cause", "data_subset_refuter"}:
        # invariância: NEW_EFFECT deveria ser PRÓXIMO de ESTIMATED_EFFECT.
        # p < 0.05 quer dizer a diferença é estatisticamente significativa,
        # ou seja, o estimador NÃO é robusto → FALHA.
        # p >= 0.05 → estimador é robusto → passa.
        p = summary.get("p_value")
        if p is None:
            return None
        return p >= 0.05

    if refuter_name == "add_unobserved_common_cause":
        # Sensitivity test — não há p-value, então não há veredicto binário.
        return None

    # Refuters não reconhecidos: tenta usar p-value se houver.
    p = summary.get("p_value")
    if p is not None:
        return p >= 0.05
    return None


# ----------------------------------------------------------------------
# Coerção de tipos
# ----------------------------------------------------------------------

def _coerce_model_df(df: pd.DataFrame, cols: list) -> Tuple[pd.DataFrame, dict]:
    """Coerce DataFrame para tipos numéricos, lidando com colunas categóricas.

    Mudanças em relação à versão anterior:
    - Aplica LabelEncoder em colunas string ANTES de ``to_numeric``.
      A versão anterior fazia direto ``to_numeric(..., errors="coerce")``
      sobre coluna string ("Pre"/"Post") → todos viravam NaN → ``dropna()``
      removia 100% das linhas. Catastrófico para CombinedDF com colunas
      como INFERRED_MENOPAUSAL_STATE.
    - Retorna debug com colunas codificadas, para auditoria.
    - Faz fillna seletivo: 0 para colunas que parecem pathway (inteiros
      não-negativos com mínimo 0), mediana para o resto. Evita perder
      linhas inteiras quando um único valor está faltando.
    """
    debug: dict = {}
    cols_present = [c for c in cols if c in df.columns]
    out = df[cols_present].copy()
    debug["columns_requested"] = list(cols)
    debug["columns_present"] = list(cols_present)
    debug["columns_missing"] = [c for c in cols if c not in df.columns]

    # Identifica colunas não-numéricas e codifica.
    encoded: list = []
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        converted = pd.to_numeric(out[c], errors="coerce")
        frac_numeric = float(converted.notna().mean())
        if frac_numeric >= 0.90:
            # Coluna predominantemente numérica com poucos NaN.
            out[c] = converted
        else:
            # Coluna genuinamente categórica → LabelEncoder.
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                out[c] = le.fit_transform(out[c].astype(str))
                encoded.append(c)
            except Exception:
                # Sem sklearn: cai para to_numeric, perdendo as linhas.
                out[c] = converted
    debug["categorical_encoded"] = encoded

    out = out.apply(pd.to_numeric, errors="coerce")

    # Fill seletivo: pathways (inteiros 0..N pequenos) → 0; resto → mediana.
    for c in out.columns:
        col_series = out[c].dropna()
        if col_series.empty:
            out[c] = out[c].fillna(0.0)
            continue
        try:
            is_int = (col_series % 1 == 0).all()
            if is_int and col_series.min() == 0 and col_series.max() < 1000:
                out[c] = out[c].fillna(0)
                continue
        except Exception:
            pass
        out[c] = out[c].fillna(col_series.median())

    out = out.fillna(0.0)
    debug["final_shape"] = out.shape
    return out, debug


# ----------------------------------------------------------------------
# Iterator de pares (treatment, outcome)
# ----------------------------------------------------------------------

def _pair_iterator(
    treatments: list,
    outcomes: list,
    all_pairs: bool,
) -> Iterator[Tuple[str, str]]:
    """Gera pares (treatment, outcome) conforme o modo escolhido.

    Mudança em relação à versão anterior: os dois ramos do ``if all_pairs``
    eram IDÊNTICOS — bug que tornava o flag inerte. Agora:

    - ``all_pairs=True``: produto cartesiano (cada treatment × cada outcome,
      com t != o).
    - ``all_pairs=False``: pareamento posicional (treatments[i] com
      outcomes[i]), truncando no comprimento menor da duas listas. Útil
      quando o usuário tem um par específico em mente sem testar todas
      as combinações.
    """
    if all_pairs:
        for t in treatments:
            for o in outcomes:
                if t != o:
                    yield t, o
    else:
        for t, o in zip(treatments, outcomes):
            if t != o:
                yield t, o


# ----------------------------------------------------------------------
# JSON com fallback para tipos numpy/pandas
# ----------------------------------------------------------------------

def _json_dump(obj: Any, path: Path) -> None:
    """Wrapper sobre json.dumps que tolera tipos numpy/pandas via ``default=str``.

    A versão anterior chamava json.dumps sem default; estimate_params com
    valores ``np.int64`` (vindos do ParameterSchema kind="int") causavam
    ``TypeError``.
    """
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

# Threshold mínimo de linhas para tentar estimação. Configurável via env-var
# (por enquanto, constante).
_MIN_ROWS_FOR_ESTIMATION = 20


# Conjunto de parâmetros que o DoWhy aceita como kwargs DIRETOS de
# ``estimate_effect`` (não em ``method_params``). Passar qualquer um destes
# via ``method_params`` causa ``TypeError: got multiple values for keyword
# argument`` porque o DoWhy internamente faz ``**method_params`` no construtor
# do estimator, que já tem esses parâmetros nomeados.
#
# Lista derivada de ``CausalEstimator.__init__`` no source do DoWhy:
# https://github.com/py-why/dowhy/blob/main/dowhy/causal_estimator.py
_ESTIMATE_DIRECT_KWARGS = frozenset({
    "control_value",
    "treatment_value",
    "test_significance",
    "evaluate_effect_strength",
    "confidence_intervals",
    "target_units",
    "effect_modifiers",
    "num_null_simulations",
    "num_simulations",
    "sample_size_fraction",
    "confidence_level",
    "need_conditional_estimates",
})


def _split_estimate_params(params: dict) -> Tuple[dict, dict]:
    """Separa params em ``(direct_kwargs, method_params)``.

    Estimators do DoWhy têm dois tipos de parâmetros:

    1. **Kwargs diretos** de ``estimate_effect``: ``test_significance``,
       ``target_units``, ``confidence_intervals``, ``control_value``,
       ``treatment_value``, etc. Passados como ``**kwargs``.

    2. **Method-specific params** (``method_params={...}``): parâmetros
       específicos de cada estimator, como ``glm_family`` (GLM), ``num_strata``
       (propensity score stratification), ``weighting_scheme`` (propensity
       weighting), ``init_params`` / ``fit_params`` (econml DML), etc.

    A versão anterior do service jogava TUDO em ``method_params``, o que
    funcionava para alguns estimators (que ignoravam silenciosamente kwargs
    desconhecidos) mas falhava em outros com ``TypeError: got multiple values
    for keyword argument`` quando o parâmetro coincidia com um do construtor.
    """
    direct: dict = {}
    method_specific: dict = {}
    for key, value in (params or {}).items():
        if key in _ESTIMATE_DIRECT_KWARGS:
            direct[key] = value
        else:
            method_specific[key] = value
    return direct, method_specific


def _run_inference_impl(
    config: InferenceRunConfig,
    progress_callback: ProgressCallback = None,
) -> dict:
    """Implementação interna do pipeline. NÃO chame diretamente do processo
    principal do Streamlit — use ``run_inference`` (o wrapper) em vez disso.

    Esta função é exportada como entry point do subprocesso filho que isola
    o trabalho da JVM do JPype/py-tetrad. Ver o docstring de ``run_inference``
    para o porquê.

    Executa o pipeline de inferência causal: identify → estimate → refute.

    Para cada par (treatment, outcome):
      1. Constrói model_df coergindo tipos (com LabelEncoder em categóricas).
      2. Valida pré-condições: treatment/outcome no grafo, linhas suficientes.
      3. Identifica o efeito; se não identificável, marca como skipped.
      4. Estima o efeito; se falhar, captura e marca como erro.
      5. Para cada refuter selecionado: tenta executar, captura exceções.

    Pares pulados ou com erro ficam em ``summary["skipped"]`` para o page
    poder reportar ao usuário.
    """
    _emit(progress_callback, f"Iniciando inferência causal: método='{config.estimate_method}'.", 0.02)

    _emit(progress_callback, f"Lendo dataset de {config.dataset_path.name}.", 0.06)
    df = _read_dataset(config.dataset_path)
    _log(f"Dataset shape: {df.shape}")

    _emit(progress_callback, f"Lendo e canonizando grafo de {config.graph_path.name}.", 0.10)
    graph_obj, dot, graph_debug = _graph_to_dot(config.graph_path, progress_callback=progress_callback)
    graph_vars = _vars_from_graph(graph_obj)
    _log(f"Variáveis no grafo: {len(graph_vars)} | arestas direcionadas: {graph_debug.get('directed_edges')}")

    out_dir = config.paths.causal_results_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_used.dot").write_text(dot, encoding="utf-8")
    _json_dump(graph_debug, out_dir / "graph_debug.json")

    # Pares a avaliar.
    pairs = list(_pair_iterator(
        config.treatment_columns, config.outcome_columns, config.all_pairs,
    ))
    if not pairs:
        _emit(
            progress_callback,
            "AVISO | Nenhum par (treatment, outcome) válido foi gerado. Verifique "
            "as seleções (todas iguais? listas vazias?) e o flag 'all_pairs'.",
            None,
        )
    else:
        _emit(progress_callback, f"Avaliando {len(pairs)} par(es) (treatment, outcome).", 0.15)

    summary: dict = {
        "pairs": [],
        "skipped": [],
        "result_dir": str(out_dir),
        "graph_debug": graph_debug,
        "estimate_method": config.estimate_method,
    }

    total_pairs = max(1, len(pairs))

    for pair_idx, (treatment, outcome) in enumerate(pairs, start=1):
        pair_progress = 0.20 + (0.70 * (pair_idx - 1) / total_pairs)
        _emit(
            progress_callback,
            f"[{pair_idx}/{len(pairs)}] Par treatment='{treatment}' outcome='{outcome}'.",
            pair_progress,
        )

        pair_slug = re.sub(r'[^a-zA-Z0-9_.-]+', '_', f"T={treatment}__Y={outcome}")
        pair_dir = out_dir / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Validação 1: treatment e outcome devem estar no grafo. Sem isso
        # o DoWhy não tem como aplicar o critério backdoor — o efeito
        # estimado seria desconectado de qualquer estrutura causal.
        if treatment not in graph_vars or outcome not in graph_vars:
            missing = [v for v in [treatment, outcome] if v not in graph_vars]
            reason = f"treatment_or_outcome_not_in_graph: missing={missing}"
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(reason, encoding="utf-8")
            continue

        # Validação 2: construir o model_df com coerção robusta.
        needed = sorted(set(graph_vars + [treatment, outcome]))
        model_df, coerce_debug = _coerce_model_df(df, needed)
        _json_dump(coerce_debug, pair_dir / "coerce_debug.json")

        if (
            len(model_df) < _MIN_ROWS_FOR_ESTIMATION
            or treatment not in model_df.columns
            or outcome not in model_df.columns
        ):
            reason = (
                f"insufficient_data: rows={len(model_df)}, "
                f"has_treatment={treatment in model_df.columns}, "
                f"has_outcome={outcome in model_df.columns}"
            )
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(reason, encoding="utf-8")
            continue

        # Pipeline DoWhy: build → identify → (validar) → estimate → refute.
        try:
            model = CausalModel(
                data=model_df, treatment=treatment, outcome=outcome, graph=graph_obj,
            )
        except Exception as exc:
            reason = f"causal_model_init_failed: {type(exc).__name__}: {exc}"
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(
                f"{reason}\n\n{traceback.format_exc()}", encoding="utf-8",
            )
            continue

        try:
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
        except Exception as exc:
            reason = f"identify_failed: {type(exc).__name__}: {exc}"
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(
                f"{reason}\n\n{traceback.format_exc()}", encoding="utf-8",
            )
            continue

        (pair_dir / "identified_estimand.txt").write_text(str(estimand), encoding="utf-8")

        # Detecta identificação vazia: estimar nesse caso lança erro opaco.
        if not _estimand_has_identification(estimand):
            reason = "effect_not_identifiable: no valid adjustment set found"
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
                "estimand_text_excerpt": str(estimand)[:500],
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(reason, encoding="utf-8")
            continue

        # Estimação.
        # Separa params em kwargs diretos vs method_params. Veja docstring
        # de ``_split_estimate_params`` para o motivo da separação.
        direct_kwargs, method_specific = _split_estimate_params(config.estimate_params)
        try:
            estimate = model.estimate_effect(
                estimand,
                method_name=config.estimate_method,
                method_params=method_specific if method_specific else None,
                **direct_kwargs,
            )
        except Exception as exc:
            reason = f"estimate_failed: {type(exc).__name__}: {exc}"
            summary["skipped"].append({
                "treatment": treatment, "outcome": outcome,
                "reason": reason, "pair_dir": str(pair_dir),
            })
            _emit(progress_callback, f"  ↳ pulado: {reason}", None)
            (pair_dir / "skipped.txt").write_text(
                f"{reason}\n\n{traceback.format_exc()}", encoding="utf-8",
            )
            continue

        # Extração estruturada do estimate (em vez de só str()).
        estimate_summary = _extract_estimate_summary(estimate)
        estimate_value = estimate_summary.get("estimate_value")
        backdoor_vars = _extract_backdoor_vars(estimand)

        estimate_payload = {
            "treatment": treatment,
            "outcome": outcome,
            "estimate_method": config.estimate_method,
            "estimate_params": config.estimate_params,
            "backdoor_vars": backdoor_vars,
            **estimate_summary,  # estimate_value, realized_expression, etc.
        }
        _json_dump(estimate_payload, pair_dir / "estimate_metadata.json")
        (pair_dir / "estimate.txt").write_text(estimate_summary["raw_text"], encoding="utf-8")

        # Refutação — agora cada refuter retorna estrutura interpretável.
        refuter_summaries: list = []
        for refuter in config.refuters:
            ref_entry: dict = {"refuter_name": refuter, "error": None}
            try:
                refutation = model.refute_estimate(
                    estimand,
                    estimate,
                    method_name=refuter,
                    **config.refuter_params.get(refuter, {}),
                )
                ref_summary = _extract_refutation_summary(refutation, refuter)
                ref_entry.update(ref_summary)
                (pair_dir / f"refuter__{refuter}.txt").write_text(
                    ref_summary["raw_text"], encoding="utf-8",
                )
                _json_dump(
                    {k: v for k, v in ref_summary.items() if k != "raw_text"},
                    pair_dir / f"refuter__{refuter}.json",
                )
            except Exception as exc:
                ref_entry["error"] = f"{type(exc).__name__}: {exc}"
                ref_entry["passes"] = None
                (pair_dir / f"refuter__{refuter}.txt").write_text(
                    f"ERROR: {ref_entry['error']}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
            refuter_summaries.append(ref_entry)

        summary["pairs"].append({
            "treatment": treatment,
            "outcome": outcome,
            "pair_dir": str(pair_dir),
            "estimate_method": config.estimate_method,
            "estimate_value": estimate_value,
            "backdoor_vars": backdoor_vars,
            "realized_expression": estimate_summary.get("realized_expression"),
            "target_units": estimate_summary.get("target_units"),
            "confidence_intervals": estimate_summary.get("confidence_intervals"),
            "significance_test": estimate_summary.get("significance_test"),
            "refuters": refuter_summaries,
        })

        # Resumo "humano" para o progress log.
        n_refuters_total = len(refuter_summaries)
        n_refuters_passed = sum(1 for r in refuter_summaries if r.get("passes") is True)
        n_refuters_failed = sum(1 for r in refuter_summaries if r.get("passes") is False)
        n_refuters_errored = sum(1 for r in refuter_summaries if r.get("error") is not None)
        _emit(
            progress_callback,
            (
                f"  ↳ par concluído. estimate={estimate_value} | "
                f"refuters: {n_refuters_passed} passou, "
                f"{n_refuters_failed} falhou, "
                f"{n_refuters_errored} erro, "
                f"{n_refuters_total - n_refuters_passed - n_refuters_failed - n_refuters_errored} sem critério."
            ),
            None,
        )

        # Força liberação de objetos pesados (CausalModel, estimate, refutations,
        # bootstrap samples) antes do próximo par. Ajuda a reduzir a pressão de
        # alocação sobre libarrow.so / glibc que, em combinação com a JVM viva
        # do JPype, pode causar SIGSEGV em ARM64. Custo: alguns ms por par.
        del model
        del estimand
        try:
            del estimate  # type: ignore[has-type]
        except NameError:
            pass
        gc.collect()

    _json_dump(summary, out_dir / "summary.json")

    _emit(
        progress_callback,
        f"Inferência concluída: {len(summary['pairs'])} par(es) com sucesso, "
        f"{len(summary['skipped'])} pulado(s).",
        1.0,
    )
    return summary


# ======================================================================
# Wrapper de subprocesso para isolamento da JVM
# ======================================================================
#
# A versão anterior chamava ``_run_inference_impl`` direto no processo do
# Streamlit. Isso causava SIGSEGV intermitente em libarrow.so depois de
# alguns minutos, com este perfil:
#
#   C [libarrow.so.1601+0xf3fd14] _Hashtable::_M_find_before_node
#   Current thread: JavaThread "Thread-3" [_thread_in_native]
#
# Diagnóstico: o JPype, usado pelo py-tetrad no passo de descoberta,
# mantém a JVM viva pelo processo todo (não pode ser reiniciada). Threads
# criadas pelo Streamlit (Tornado IO/WebSocket) acabam sendo
# auto-attached à JVM quando entram em código Python que toca alguma
# global do JPype. Quando essas threads JVM tocam pyarrow.compute (via
# pandas, dowhy, etc.) concorrentemente com a thread main, há race no
# function registry global do Arrow (ARROW-10784) → corrupção de memória
# → SIGSEGV no momento de uma busca subsequente.
#
# Tentativas anteriores que não funcionaram:
# - Limitar threads via env (OMP_NUM_THREADS=1, etc.): tornava o pipeline
#   lento demais para uso prático.
# - pyarrow_compat (inicialização eager + set_cpu_count(1)): não isolou
#   da thread JVM. Crash voltou.
# - Pin de pyarrow<17: mitigou o jemalloc bg thread, mas a race no
#   FunctionRegistry persiste em todas as versões.
# - shutdownJVM no fim da descoberta: JPype não suporta shutdown fora da
#   main thread + JVM não pode ser reiniciada → solução inviável aqui.
#
# Solução escolhida: rodar a inferência num **subprocesso filho** spawnado
# com ``multiprocessing.get_context("spawn")``. O filho:
#   - Nasce fresco, sem JVM viva, sem threads attached.
#   - Importa pandas/dowhy/pyarrow do zero (custo: ~3-5s de startup).
#   - Executa o pipeline completo.
#   - Reporta progresso via fila multiprocessing.
#   - Termina, e seu state nativo é descartado pelo SO.
#
# Custo extra: ~3-5s de import a cada inferência. Aceitável dado que a
# inferência leva minutos. Benefício: zero crash possível em libarrow,
# porque não há JVM no subprocesso.


_PROGRESS_SENTINEL = "__PROGRESS_END__"
_RESULT_SENTINEL = "__RESULT_BEGIN__"
_ERROR_SENTINEL = "__ERROR_BEGIN__"


def _subprocess_entry(config_pickle: bytes, queue: Any) -> None:
    """Entry point do subprocesso filho.

    Recebe a config serializada (pickle) e uma fila multiprocessing.
    Decodifica a config, define um callback que escreve mensagens na fila,
    e chama ``_run_inference_impl``. No fim, escreve o resultado (ou erro)
    na fila e termina.

    Isolamento garantido: este processo nasce SEM JVM e SEM threads do
    pai. Tudo aqui é fresco.
    """
    import pickle
    import traceback as _tb

    try:
        config = pickle.loads(config_pickle)
    except Exception as exc:
        queue.put((_ERROR_SENTINEL, f"Failed to unpickle config: {exc}", _tb.format_exc()))
        return

    def _child_progress(message: str, progress: Optional[float] = None) -> None:
        # Escreve na fila e segue. O pai lê e replica para o callback de UI.
        try:
            queue.put(("progress", message, progress))
        except Exception:
            # Se a fila quebrou (pai morreu?), apenas ignore.
            pass

    try:
        result = _run_inference_impl(config, progress_callback=_child_progress)
        queue.put((_RESULT_SENTINEL, result))
    except Exception as exc:
        queue.put((_ERROR_SENTINEL, f"{type(exc).__name__}: {exc}", _tb.format_exc()))
    finally:
        # Sentinela final para o pai saber que o filho terminou (mesmo em
        # caso de exception).
        try:
            queue.put((_PROGRESS_SENTINEL,))
        except Exception:
            pass


def run_inference(
    config: InferenceRunConfig,
    progress_callback: ProgressCallback = None,
) -> dict:
    """Executa o pipeline de inferência causal em subprocesso isolado.

    Esta função é o **ponto de entrada público**. Ela:

    1. Spawna um subprocesso Python fresco (sem JVM, sem state do pai).
    2. Passa a ``config`` serializada e uma fila de progresso.
    3. Lê mensagens da fila em loop, replicando para o ``progress_callback``
       (que pode tocar ``st.session_state`` no pai sem problemas).
    4. Quando o filho termina, lê o resultado da fila e retorna.

    Em caso de crash do filho (SIGSEGV ou outro), captura o exit code e
    levanta ``RuntimeError`` com detalhes — em vez de derrubar o Streamlit.

    Veja o comentário acima sobre o porquê desta arquitetura.
    """
    import multiprocessing
    import pickle

    _emit(progress_callback, "Inicializando subprocesso isolado para inferência...", 0.01)

    # spawn (vs. fork): garante Python fresco, sem herdar state do Streamlit.
    # Essencial para evitar herdar a JVM viva e suas threads attached.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    try:
        config_pickle = pickle.dumps(config)
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao serializar config para subprocesso: {exc}"
        ) from exc

    process = ctx.Process(
        target=_subprocess_entry,
        args=(config_pickle, queue),
        daemon=False,
    )
    process.start()
    _emit(progress_callback, f"Subprocesso iniciado (PID {process.pid}).", 0.02)

    result: Optional[dict] = None
    error: Optional[tuple] = None

    # Loop de leitura da fila. Sai quando recebe o sentinela final ou
    # quando o filho morre (crash).
    while True:
        try:
            # Timeout curto: permite detectar crash via process.is_alive().
            message = queue.get(timeout=2.0)
        except Exception:
            # Timeout — verifica se o filho ainda está vivo.
            if not process.is_alive():
                # Filho morreu sem mandar sentinela. Provável SIGSEGV.
                exit_code = process.exitcode
                error = (
                    f"Subprocesso terminou inesperadamente (exit code={exit_code}).",
                    (
                        "Isso geralmente indica SIGSEGV nativo (libarrow, JVM ou similar). "
                        "Verifique os logs do container para detalhes."
                    ),
                )
                break
            # Filho vivo, só está demorando. Continua esperando.
            continue

        kind = message[0]
        if kind == "progress":
            _, msg, prog = message
            if progress_callback is not None:
                progress_callback(msg, prog)
        elif kind == _RESULT_SENTINEL:
            result = message[1]
        elif kind == _ERROR_SENTINEL:
            error = (message[1], message[2] if len(message) > 2 else "")
        elif kind == _PROGRESS_SENTINEL:
            # Sentinela final — filho terminou normalmente. Sai do loop.
            break

    # Junta o subprocesso (libera recursos).
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)

    if error is not None:
        msg, tb = error
        raise RuntimeError(f"Inferência falhou no subprocesso: {msg}\n\n{tb}")

    if result is None:
        raise RuntimeError(
            "Subprocesso terminou sem retornar resultado e sem erro explícito. "
            "Provavelmente crash nativo. Verifique os logs."
        )

    return result