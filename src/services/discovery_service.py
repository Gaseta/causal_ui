from __future__ import annotations

import importlib
import inspect
import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.models import DiscoveryRunConfig
from src.services.graph_utils import dot_from_edges, save_dot_string_as_png


# ----------------------------------------------------------------------
# Tipos auxiliares
# ----------------------------------------------------------------------

ProgressCallback = Optional[Callable[[str, Optional[float]], None]]


# Tokens de relação suportados em strings de aresta no formato TETRAD.
# Cada entry mapeia o token para uma descrição usada no DOT.
TETRAD_REL_TOKENS = ("-->", "<--", "---", "o->", "<-o", "o-o")


# ----------------------------------------------------------------------
# Logging e progresso
# ----------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[discovery] {msg}", flush=True)


def _emit(progress_callback: ProgressCallback, msg: str, progress: Optional[float] = None) -> None:
    """Emite uma mensagem para o callback de UI E para o stdout do servidor.

    Use para marcos visíveis para o usuário. Para logs de alta frequência, use
    apenas ``_log`` (não polui o histórico da UI).
    """
    _log(msg)
    if progress_callback is not None:
        progress_callback(msg, progress)


# ----------------------------------------------------------------------
# Carregamento e preparação dos dados
# ----------------------------------------------------------------------

# Heurística para identificar a primeira coluna como ID de paciente/amostra.
# Aplicada por ``_ensure_dataframe`` para decidir se carrega com index_col=0.
_ID_COLUMN_PATTERNS = re.compile(
    r"^(patient[_ ]?id|sample[_ ]?id|id|case[_ ]?id|tcga|mb|p\d+|s\d+)",
    flags=re.IGNORECASE,
)


def _looks_like_id_column(series_head: pd.Series) -> bool:
    """Heurística leve: a coluna parece conter identificadores de paciente/amostra.

    Critérios cumulativos:
      - Todos os valores são únicos no head amostrado.
      - Os valores são string-like (ou conversíveis).
      - O nome da primeira coluna casa com padrões conhecidos (PATIENT_ID, SAMPLE_ID,
        TCGA-*, MB-*, P\\d+, etc.).
    """
    name = str(series_head.name or "")
    if not _ID_COLUMN_PATTERNS.match(name):
        # Mesmo sem nome bater, valores únicos string-like com formato uniforme
        # também são suspeitos. Aceita se todos os valores parecem códigos.
        sample = series_head.dropna().astype(str).tolist()
        if not sample:
            return False
        # Todos têm o mesmo formato (mesmo número de hífens, mesmo prefixo) é
        # mais forte do que só "todos únicos".
        prefixes = {s.split("-")[0] if "-" in s else s[:3] for s in sample}
        if len(prefixes) > 2:
            return False
    return series_head.is_unique


def _ensure_dataframe(path: Path) -> pd.DataFrame:
    """Carrega o dataset detectando automaticamente se a 1ª coluna é um índice (ID).

    Mudança em relação à versão anterior: a detecção do ``index_col`` é feita
    agora pela natureza da primeira coluna (heurística ``_looks_like_id_column``),
    não pelo formato do arquivo. Isso resolve dois problemas:
      (a) CSV com PATIENT_ID na 1ª coluna era lido como coluna comum.
      (b) TSV onde a 1ª coluna era uma feature genuína era erroneamente
          virada índice por ``index_col=0``.
    """
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    # Lê sem index_col para inspecionar.
    probe = pd.read_csv(path, sep=sep, low_memory=False, nrows=50)
    if probe.shape[1] >= 2 and _looks_like_id_column(probe.iloc[:, 0]):
        return pd.read_csv(path, sep=sep, low_memory=False, index_col=0)
    return pd.read_csv(path, sep=sep, low_memory=False)


def _is_pathway_like_column(column_name: str) -> bool:
    """Heurística para detectar colunas de pathway no CombinedDF.

    Pathways do Reactome saem do nosso pré-processamento com formato
    ``"<Root>"`` (super) ou ``"<Root> -- <Sub>"`` (sub). O separador ``" -- "``
    é o marcador mais confiável.

    Para super-pathways de uma palavra (e.g. "Autophagy", "Metabolism"), não
    temos um marcador único — o detect cai no fallback estatístico (valores
    inteiros não-negativos com mínimo zero) em ``_prepare_numeric_matrix``.
    """
    return " -- " in str(column_name)


def _prepare_numeric_matrix(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
) -> Tuple[np.ndarray, list, dict]:
    """Prepara a matriz numérica para os algoritmos de descoberta causal.

    Distingue entre colunas clínicas (NaN → mediana) e colunas de pathway
    (NaN → 0). A distinção é feita por duas heurísticas combinadas:
      1. Nomes com " -- " (subpathway Reactome) são pathways.
      2. Colunas com valores inteiros não-negativos cujo mínimo é zero também
         são tratadas como pathways (cobre super-pathways de uma palavra).

    Outras transformações: codificação categórica (LabelEncoder), conversão
    numérica tolerante (90% de cobertura é suficiente para tentar converter),
    e ruído microscópico em colunas com variância zero para evitar singularidades
    nos testes de independência.
    """
    debug: dict = {}
    out = df.copy()
    # 1) Remoção de IDs.
    dropped = [c for c in ["PATIENT_ID", "SAMPLE_ID"] if c in out.columns]
    if dropped:
        out = out.drop(columns=dropped)
    debug["dropped_id_columns"] = dropped

    # 2) Classificação prévia de colunas como "pathway-like" pelo nome.
    pathway_cols_by_name: list = [c for c in out.columns if _is_pathway_like_column(c)]

    # 3) Tenta conversão numérica para cada coluna; o que sobrar é categórico.
    categorical_cols: list = []
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        converted = pd.to_numeric(out[c], errors="coerce")
        frac_numeric = float(converted.notna().mean())
        # Limite mais permissivo do que antes (0.95 → 0.90): permite que
        # colunas numéricas com pequena % de "Unknown" sejam tratadas como
        # numéricas em vez de virarem categóricas via LabelEncoder
        # (que destruiria a relação numérica).
        if frac_numeric >= 0.90:
            out[c] = converted
        else:
            categorical_cols.append(c)

    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder

        for c in categorical_cols:
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype(str))

    # 4) Garante numerização final (caso a passagem anterior tenha deixado object).
    out = out.apply(pd.to_numeric, errors="coerce")

    # 5) Detecção adicional de pathways: colunas com valores inteiros não-negativos
    #    cujo mínimo é zero. Cobre super-pathways de uma palavra.
    pathway_cols_by_dtype: list = []
    for c in out.columns:
        if c in pathway_cols_by_name:
            continue
        col_series = out[c].dropna()
        if col_series.empty:
            continue
        # Inteiros não-negativos com mínimo zero é o "shape" típico de um
        # pathway driver-count.
        try:
            is_integer_like = (col_series % 1 == 0).all()
            if is_integer_like and col_series.min() == 0 and col_series.max() < 1000:
                pathway_cols_by_dtype.append(c)
        except Exception:
            continue

    pathway_cols = sorted(set(pathway_cols_by_name + pathway_cols_by_dtype))
    clinical_cols = [c for c in out.columns if c not in pathway_cols]

    # 6) Fill diferenciado: pathways recebem 0, clínicos recebem mediana.
    if pathway_cols:
        out[pathway_cols] = out[pathway_cols].fillna(0)
    if clinical_cols:
        out[clinical_cols] = out[clinical_cols].fillna(
            out[clinical_cols].median(numeric_only=True)
        )
    out = out.fillna(0.0).astype(float)

    # 7) Ruído microscópico em colunas com variância zero para evitar
    #    singularidades em testes de independência baseados em correlação.
    zero_var = list(out.columns[out.std(axis=0, ddof=0) == 0.0])
    if zero_var:
        rng = np.random.default_rng(42)
        for c in zero_var:
            out[c] = out[c] + rng.normal(0.0, 1e-6, size=len(out))

    # 8) Remoção de colunas linearmente dependentes (multicolinearidade).
    #    Sem isso, a matriz de correlação fica singular e testes de
    #    independência baseados em correlação parcial (Fisher-Z do PC, por
    #    exemplo) falham com "Singular matrix". É comum em datasets de
    #    pathway porque alguns pathways do Reactome são alias um do outro
    #    (e.g. Generic Transcription Pathway ≡ RNA Polymerase II Transcription
    #    em datasets de painel direcionado).
    out, collinearity_debug = _remove_collinear_columns(out, progress_callback=progress_callback)

    debug.update({
        "final_shape": out.shape,
        "categorical_encoded": categorical_cols,
        "pathway_cols_by_name": pathway_cols_by_name,
        "pathway_cols_by_dtype": pathway_cols_by_dtype,
        "pathway_cols_total": len(pathway_cols),
        "clinical_cols_total": len(clinical_cols),
        "final_columns": out.columns.tolist(),
        "zero_variance_cols": zero_var,
        "collinearity": collinearity_debug,
    })

    if progress_callback is not None:
        n_removed = (
            len(collinearity_debug.get("duplicates_removed", []))
            + len(collinearity_debug.get("rank_deficient_removed", []))
        )
        progress_callback(
            f"Preparada matriz {out.shape[0]} × {out.shape[1]} "
            f"(pathways={len(pathway_cols)}, clínicas={len(clinical_cols)}, "
            f"categóricas codificadas={len(categorical_cols)}, "
            f"colunas redundantes removidas={n_removed}).",
            None,
        )

    return out.to_numpy(dtype=float), out.columns.tolist(), debug


def _remove_collinear_columns(
    df: pd.DataFrame,
    corr_threshold: float = 0.999,
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, dict]:
    """Remove colunas linearmente dependentes para evitar matriz de correlação singular.

    Multicolinearidade perfeita (ou quase) faz testes de independência baseados em
    correlação parcial (Fisher-Z, padrão do PC) falharem com "Singular matrix".
    Isso é comum em datasets de pathway porque alguns pathways do Reactome
    são alias um do outro — por exemplo, ``Generic Transcription Pathway`` e
    ``RNA Polymerase II Transcription`` compartilham praticamente os mesmos
    genes na hierarquia, e suas colunas no CombinedDF acabam idênticas.

    Estratégia em duas fases:

    1. **Duplicatas exatas/quase exatas** (``|corr| >= corr_threshold``):
       agrupa as colunas em "clusters" de duplicatas, mantém apenas uma
       representante por cluster. Critério para escolher a que fica: maior
       variância (mais informativa); em empate, a alfabeticamente menor
       (determinístico).

    2. **Rank deficiência residual**: se mesmo após o passo 1 a matriz ainda
       não tem rank cheio, remove colunas iterativamente via decomposição
       QR com pivotamento, descartando aquelas cujo pivô tem norma
       desprezível.

    Retorna um novo DataFrame contendo apenas as colunas mantidas, junto com
    um dicionário ``debug`` listando o que foi removido e por quê.
    """
    debug: dict = {
        "duplicates_removed": [],
        "rank_deficient_removed": [],
        "duplicate_groups": [],
    }

    out = df.copy()
    if out.shape[1] < 2:
        return out, debug

    # --- Fase 1: duplicatas via correlação ---
    # Trabalhamos com dtype float e variância > 0 (corr exige).
    out_numeric = out.select_dtypes(include=[np.number])
    if out_numeric.shape[1] >= 2:
        corr = out_numeric.corr().abs().fillna(0.0)
        cols = out_numeric.columns.tolist()
        variances = out_numeric.var(numeric_only=True).fillna(0.0)

        # Construção dos clusters de colunas "iguais" via union-find ingênuo.
        parent = {c: c for c in cols}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= corr_threshold:
                    _union(cols[i], cols[j])

        # Agrupa por raiz e escolhe representante de cada grupo.
        groups: dict = {}
        for c in cols:
            groups.setdefault(_find(c), []).append(c)

        to_drop: list = []
        for root, members in groups.items():
            if len(members) <= 1:
                continue
            # Critério: maior variância primeiro; empate por ordem alfabética
            # para determinismo. A primeira da lista ordenada é a "mantida".
            members_sorted = sorted(members, key=lambda c: (-variances.get(c, 0.0), c))
            keep = members_sorted[0]
            drop = members_sorted[1:]
            to_drop.extend(drop)
            debug["duplicate_groups"].append({
                "kept": keep,
                "dropped": drop,
                "max_pair_corr": float(
                    corr.loc[members, members].where(
                        ~np.eye(len(members), dtype=bool)
                    ).max().max()
                ),
            })

        if to_drop:
            out = out.drop(columns=to_drop)
            debug["duplicates_removed"] = to_drop
            if progress_callback is not None:
                progress_callback(
                    f"Multicolinearidade: removidas {len(to_drop)} coluna(s) duplicada(s)/quase-duplicada(s) "
                    f"(|corr|≥{corr_threshold}). Mantida(s): {len(debug['duplicate_groups'])} representante(s).",
                    None,
                )

    # --- Fase 2: rank deficiência residual ---
    out_numeric = out.select_dtypes(include=[np.number])
    if out_numeric.shape[1] >= 2:
        matrix = out_numeric.to_numpy(dtype=float)
        # Centra para checar dependência linear (sem viés de offset).
        matrix_centered = matrix - matrix.mean(axis=0)
        rank = int(np.linalg.matrix_rank(matrix_centered))
        n_cols = matrix_centered.shape[1]

        if rank < n_cols:
            # Decomposição QR com pivotamento revela quais colunas são
            # combinações lineares das outras. Pivôs com |R[i,i]| pequeno
            # indicam colunas redundantes.
            try:
                from scipy.linalg import qr as scipy_qr
                Q, R, piv = scipy_qr(matrix_centered, mode="economic", pivoting=True)
                # Tolerância relativa proporcional ao maior pivô.
                diag = np.abs(np.diag(R))
                tol = max(matrix_centered.shape) * np.finfo(float).eps * diag.max()
                # Pivôs além do rank são redundantes.
                redundant_idx = piv[rank:]
            except ImportError:
                # Fallback: usa numpy.linalg.qr sem pivotamento (menos preciso,
                # mas funciona em ambientes sem scipy).
                Q, R = np.linalg.qr(matrix_centered, mode="reduced")
                diag = np.abs(np.diag(R))
                # Sem pivotamento, os "menores" pivôs são candidatos.
                redundant_idx = np.argsort(diag)[: n_cols - rank]

            redundant_cols = [out_numeric.columns[i] for i in redundant_idx]
            if redundant_cols:
                out = out.drop(columns=redundant_cols)
                debug["rank_deficient_removed"] = redundant_cols
                if progress_callback is not None:
                    progress_callback(
                        f"Multicolinearidade residual: removidas {len(redundant_cols)} coluna(s) "
                        f"linearmente dependente(s) via QR.",
                        None,
                    )

    return out, debug


# ----------------------------------------------------------------------
# Conversão de grafos para strings TETRAD
# ----------------------------------------------------------------------

# Regex que casa APENAS com nomes genéricos do tipo X<n> (e.g. "X0", "X12").
# Diferente da versão anterior, que casava com qualquer "(\d+)$" — o que fazia
# "BRCA1" ser mapeado para variable_names[1] erroneamente.
_GENERIC_NODE_NAME = re.compile(r"^X(\d+)$")


def _detect_node_indexing(node_names: list) -> int:
    """Detecta a base de indexação dos nomes genéricos no grafo (0 ou 1).

    Verifica se algum nó se chama "X0": se sim, é 0-indexed (causal-learn).
    Caso contrário, presume 1-indexed (py-tetrad / Tetrad).
    """
    for name in node_names:
        if str(name) == "X0":
            return 0
    return 1


def _edges_to_tetrad_strings(result: Any, variable_names: list) -> list:
    """Converte um grafo do causal-learn para strings no formato TETRAD."""
    from causallearn.graph.Endpoint import Endpoint

    G = result.G if hasattr(result, "G") else result["G"]
    raw_node_names = [str(e.get_node1().get_name()) for e in G.get_graph_edges()] + \
                     [str(e.get_node2().get_name()) for e in G.get_graph_edges()]
    base_index = _detect_node_indexing(raw_node_names)

    def resolve_name(node_name: str) -> str:
        s = str(node_name)
        match = _GENERIC_NODE_NAME.match(s)
        if match is None:
            # Nome não é genérico (e.g. "BRCA1", "AGE") — preservar como veio.
            return s
        idx = int(match.group(1))
        target_idx = idx if base_index == 0 else idx - 1
        if 0 <= target_idx < len(variable_names):
            return variable_names[target_idx]
        # Índice fora do range — preserva o nome bruto para diagnóstico.
        return s

    edges: list = []
    for e in G.get_graph_edges():
        n1 = resolve_name(e.get_node1().get_name())
        n2 = resolve_name(e.get_node2().get_name())
        ep1 = e.get_endpoint1()
        ep2 = e.get_endpoint2()
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            rel = "-->"
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            rel = "<--"
        elif ep1 == Endpoint.TAIL and ep2 == Endpoint.TAIL:
            rel = "---"
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.ARROW:
            rel = "o->"
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.CIRCLE:
            rel = "<-o"
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.CIRCLE:
            rel = "o-o"
        else:
            rel = "---"
        edges.append(f"{n1} {rel} {n2}")
    return edges


# ----------------------------------------------------------------------
# Parsing de strings TETRAD para uso downstream
# ----------------------------------------------------------------------

# Regex de aresta TETRAD: "<nó> <rel> <nó>", onde rel é um dos tokens
# suportados. Não usamos lazy/greedy ambíguo porque os tokens são fixos.
_TETRAD_EDGE_REGEX = re.compile(
    r"^(?P<a>.+?)\s+(?P<rel>-->|<--|---|o->|<-o|o-o)\s+(?P<b>.+?)$"
)


def _parse_tetrad_edges(edges: list) -> list:
    """Parseia strings TETRAD em tuplas ``(a, rel, b)``.

    Retorna lista de tuplas únicas (preserva ordem da primeira ocorrência),
    descartando entradas inválidas e self-loops.
    """
    parsed: list = []
    seen: set = set()
    for edge in edges:
        match = _TETRAD_EDGE_REGEX.match(str(edge).strip())
        if not match:
            continue
        a = match.group("a").strip()
        rel = match.group("rel")
        b = match.group("b").strip()
        if a == b:
            continue
        # Normaliza orientação: "<--" vira "-->" com nós invertidos
        # para deduplicação consistente.
        if rel == "<--":
            a, b, rel = b, a, "-->"
        elif rel == "<-o":
            a, b, rel = b, a, "o->"
        key = (a, b, rel)
        if key in seen:
            continue
        seen.add(key)
        parsed.append((a, rel, b))
    return parsed


def _directed_edges_from_tetrad_strings(edges: list) -> list:
    """Compatibilidade: retorna lista de tuplas (a, b) apenas para arestas
    direcionadas (``-->`` e ``<--``).

    Mantida para callers que ainda esperam o formato antigo (`dot_from_edges`).
    """
    out: list = []
    for a, rel, b in _parse_tetrad_edges(edges):
        if rel == "-->":
            out.append((a, b))
    return out


def _dot_attrs_for_relation(rel: str) -> str:
    """Atributos GraphViz que codificam visualmente o tipo de aresta TETRAD.

    O tipo de aresta é semanticamente importante em descoberta causal:
      - ``-->``: aresta direcionada (CPDAG, MAG)
      - ``---``: aresta não-direcionada (CPDAG: orientação ambígua dentro
                  da classe de equivalência)
      - ``o->`` / ``o-o``: arestas com ponta circular (PAG / FCI: representa
                  incerteza sobre confundimento latente)
    """
    if rel == "-->":
        return ""  # Default (arrowhead=normal, dir=forward) é o esperado.
    if rel == "---":
        return ' [dir=none]'
    if rel == "o->":
        return ' [dir=both, arrowtail=odot, arrowhead=normal]'
    if rel == "o-o":
        return ' [dir=both, arrowtail=odot, arrowhead=odot]'
    return ""


def _dot_from_tetrad_edges(parsed_edges: list) -> str:
    """Constrói DOT preservando todos os tipos de arestas TETRAD.

    Substitui o uso direto de ``dot_from_edges`` quando há arestas
    não-direcionadas ou ponta-circular (PC, FCI, GFCI, BFCI).
    """
    if not parsed_edges:
        return "digraph G {\n}\n"
    # Conjunto de nós.
    nodes: set = set()
    for a, _, b in parsed_edges:
        nodes.add(a)
        nodes.add(b)
    lines: list = ["digraph G {", "  rankdir=LR;"]
    for node in sorted(nodes):
        # Aspas duplas em DOT precisam ser escapadas.
        safe = str(node).replace('"', r'\"')
        lines.append(f'  "{safe}";')
    for a, rel, b in parsed_edges:
        safe_a = str(a).replace('"', r'\"')
        safe_b = str(b).replace('"', r'\"')
        attrs = _dot_attrs_for_relation(rel)
        lines.append(f'  "{safe_a}" -> "{safe_b}"{attrs};')
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Backends individuais (causal-learn)
# ----------------------------------------------------------------------

def _run_pc_with_causallearn(data_matrix: np.ndarray, params: dict):
    from causallearn.search.ConstraintBased.PC import pc

    return pc(
        data_matrix,
        indep_test=params.get("indep_test", "fisherz"),
        alpha=float(params.get("alpha", 0.05)),
        stable=bool(params.get("stable", True)),
        uc_rule=0,
        uc_priority=2,
        max_k=int(params.get("max_k", 3)),
    )


def _run_ges_with_causallearn(data_matrix: np.ndarray):
    from causallearn.search.ScoreBased.GES import ges

    return ges(data_matrix)


def _run_fci_with_causallearn(data_matrix: np.ndarray, params: dict):
    from causallearn.search.ConstraintBased.FCI import fci

    graph, edges = fci(
        data_matrix,
        independence_test_method=params.get("indep_test", "fisherz"),
        alpha=float(params.get("alpha", 0.05)),
    )
    return {"G": graph, "edges": edges}


# ----------------------------------------------------------------------
# Backend py-tetrad
# ----------------------------------------------------------------------

def _load_tetrad_search_class():
    """Carrega a classe TetradSearch da biblioteca py-tetrad.

    Aceita duas localizações do módulo (caso o usuário tenha clonado o
    repositório oficial ou instalado via pip com outro nome de pacote).
    """
    candidates = [
        ("tools.TetradSearch", "TetradSearch"),
        ("pytetrad.tools.TetradSearch", "TetradSearch"),
    ]
    errors: list = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("Could not import TetradSearch. " + " | ".join(errors))


def _coerce_param_value(value: Any) -> Any:
    """Converte tipos numpy/pandas para tipos nativos Python.

    Necessário porque alguns setters do py-tetrad chamam métodos Java via
    JPype, e Java é estrito com tipos primitivos.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _apply_params_to_object(obj: Any, params: dict) -> dict:
    """Aplica parâmetros como setters no objeto py-tetrad.

    Procura métodos com nomes em várias convenções (snake_case, camelCase,
    com prefixo set_/use_) e usa o primeiro que aceitar o valor.

    Retorna o dicionário de parâmetros efetivamente aplicados — útil para
    diagnóstico (alguns parâmetros da UI podem não existir no algoritmo).
    """
    applied: dict = {}
    if not params:
        return applied

    for key, raw_value in params.items():
        value = _coerce_param_value(raw_value)
        method_names = [
            f"set_{key}",
            key,
            f"use_{key}",
            f"set{''.join(part.capitalize() for part in key.split('_'))}",
        ]
        for name in method_names:
            method = getattr(obj, name, None)
            if not callable(method):
                continue
            try:
                method(value)
                applied[key] = value
                break
            except TypeError:
                # Algumas implementações esperam kwargs em vez de posicional.
                try:
                    method(**{key: value})
                    applied[key] = value
                    break
                except Exception:
                    continue
            except Exception:
                continue
    return applied


def _invoke_algorithm(search_obj: Any, algorithm_key: str) -> Any:
    """Invoca o algoritmo no objeto TetradSearch, sem passar params como kwargs.

    Importante: como ``_apply_params_to_object`` já configurou os parâmetros
    via setters, aqui chamamos o método de execução sem argumentos. Essa é a
    forma idiomática de uso do py-tetrad.

    A versão anterior passava parâmetros DUPLICADOS (setters + kwargs), o que
    podia gerar comportamento inconsistente entre versões da biblioteca.
    """
    candidates = {
        "pc": ["run_pc", "pc"],
        "fci": ["run_fci", "fci"],
        "ges": ["run_ges", "ges", "run_fges", "fges"],
        "fges": ["run_fges", "fges", "run_ges", "ges"],
        "gfci": ["run_gfci", "gfci"],
        "boss": ["run_boss", "boss"],
        "grasp": ["run_grasp", "grasp"],
        "bfci": ["run_bfci", "bfci"],
        "lv_lite": ["run_lv_lite", "lv_lite", "run_lvlite", "lvlite"],
    }
    names = candidates.get(algorithm_key, [f"run_{algorithm_key}", algorithm_key])
    last_error: Optional[Exception] = None

    for name in names:
        method = getattr(search_obj, name, None)
        if not callable(method):
            continue
        try:
            sig = inspect.signature(method)
            # Métodos sem parâmetros são chamados diretamente.
            if len(sig.parameters) == 0:
                return method()
            # Métodos com parâmetros: chamamos sem args, porque setters já
            # configuraram o estado. Se o método EXIGIR argumentos, ele vai
            # lançar TypeError, capturado abaixo, e tentamos a próxima opção.
            return method()
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise AttributeError(f"No callable py-tetrad method found for algorithm '{algorithm_key}'.")


def _extract_edges_from_string(text: str) -> list:
    """Extrai linhas que parecem arestas TETRAD de um texto multi-linha."""
    edges: list = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s and any(tok in s for tok in TETRAD_REL_TOKENS):
            edges.append(s)
    return edges


def _extract_pytetrad_result(
    search_obj: Any,
    result: Any,
    variable_names: list,
) -> Tuple[list, Optional[str], dict]:
    """Extrai edges (e opcionalmente DOT) do retorno do py-tetrad.

    Cobre várias formas em que diferentes versões/algoritmos do py-tetrad
    expõem o resultado:
      - dict ``{"edges": [...], "dot": "..."}``
      - lista de strings de aresta
      - string única multi-linha com o grafo
      - objeto com método ``get_graph_edges`` (igual ao causal-learn)
      - objeto sem retorno relevante: tenta métodos do ``search_obj``
        (``get_string``, ``get_dot``, etc.)
    """
    debug: dict = {"result_type": type(result).__name__}
    dot: Optional[str] = None

    # Caminho 1: dict com chave "edges".
    if isinstance(result, dict) and "edges" in result:
        edges = [str(e) for e in result["edges"]]
        return edges, result.get("dot"), debug

    # Caminho 2: lista direta de arestas.
    if isinstance(result, list):
        return [str(e) for e in result], dot, debug

    # Caminho 3: string única multi-linha.
    if isinstance(result, str):
        edges = _extract_edges_from_string(result)
        if edges:
            return edges, dot, debug

    # Caminho 4: objeto tipo "Graph" do causal-learn.
    if hasattr(result, "get_graph_edges") or hasattr(result, "G"):
        try:
            edges = _edges_to_tetrad_strings(result, variable_names)
            return edges, dot, debug
        except Exception as exc:
            debug["graph_parse_error"] = str(exc)

    # Caminho 5: o resultado em si não trouxe nada útil — interroga o
    # objeto de busca por métodos que exponham o grafo como texto.
    edges: list = []
    for attr_name in ["get_string", "get_graph_as_string", "get_string_graph", "get_result_graph_string"]:
        fn = getattr(search_obj, attr_name, None)
        if not callable(fn):
            continue
        try:
            text = fn()
            if text is not None:
                edges = _extract_edges_from_string(str(text))
                if edges:
                    debug["edges_source"] = attr_name
                    break
        except Exception as exc:
            debug[f"{attr_name}_error"] = str(exc)

    # Tenta também extrair DOT se disponível.
    for attr_name in ["get_dot", "get_dot_graph", "get_dot_string"]:
        fn = getattr(search_obj, attr_name, None)
        if not callable(fn):
            continue
        try:
            maybe_dot = fn()
            if maybe_dot:
                dot = str(maybe_dot)
                debug["dot_source"] = attr_name
                break
        except Exception as exc:
            debug[f"{attr_name}_error"] = str(exc)

    if not edges:
        raise RuntimeError("py-tetrad executed but no edges could be extracted from the result.")
    return edges, dot, debug


def _attempt_pytetrad(
    algorithm_key: str,
    prepared_df: pd.DataFrame,
    params: dict,
    variable_names: list,
) -> dict:
    """Tenta executar o algoritmo via py-tetrad.

    Recebe um DataFrame **já preparado** (numérico, sem IDs, sem NaN). Isso é
    uma mudança em relação à versão anterior, que recebia o DataFrame cru e
    passava IDs/NaN/categóricas direto para o TetradSearch (causando falhas
    silenciosas que disparavam o fallback para causal-learn sem o usuário
    perceber que py-tetrad nunca rodou de fato).
    """
    TetradSearch = _load_tetrad_search_class()
    search_obj = TetradSearch(prepared_df.copy())

    # 1) Trata `indep_test` (e seu `alpha`) como caso especial: o py-tetrad
    #    NÃO expõe um setter genérico `set_indep_test(name)`. Em vez disso
    #    usa setters explícitos por teste: `use_fisher_z`, `use_chi_square`
    #    etc. Sem esta tradução, o usuário poderia escolher "chisq" na UI
    #    e a execução cairia silenciosamente no teste default do py-tetrad
    #    (geralmente Fisher-Z). Aqui interceptamos antes do loop genérico.
    remaining_params, indep_test_applied = _apply_pytetrad_independence_test(
        search_obj, params,
    )

    # 2) Aplica os parâmetros restantes pelo loop genérico de setters.
    applied_params = _apply_params_to_object(search_obj, remaining_params)
    if indep_test_applied is not None:
        applied_params["indep_test"] = indep_test_applied

    result = _invoke_algorithm(search_obj, algorithm_key)
    edges, dot, extract_debug = _extract_pytetrad_result(search_obj, result, variable_names)
    return {
        "engine": "py-tetrad",
        "search_obj_type": type(search_obj).__name__,
        "result": result,
        "edges": edges,
        "dot": dot,
        "applied_params": applied_params,
        "debug": extract_debug,
    }


# ----------------------------------------------------------------------
# Tradução de ``indep_test`` para py-tetrad
# ----------------------------------------------------------------------

# Mapeia o nome curto do teste (vindo da UI / parameter_registry) para o
# nome do método ``use_*`` exposto pela classe ``TetradSearch`` do py-tetrad.
# Esses nomes seguem o padrão snake_case usado no source do py-tetrad
# (ver TetradSearch.py em https://github.com/cmu-phil/py-tetrad).
_PYTETRAD_INDEP_TEST_SETTERS = {
    "fisherz": "use_fisher_z",
    "chisq": "use_chi_square",
    "gsq": "use_g_square",
    "kci": "use_kci",
    # mv_fisherz não tem equivalente exato no py-tetrad — cai no Fisher-Z
    # padrão. Documentado abaixo na função.
}


def _apply_pytetrad_independence_test(
    search_obj: Any,
    params: dict,
) -> Tuple[dict, Optional[str]]:
    """Aplica ``indep_test`` no search_obj usando o setter correto do py-tetrad.

    Retorna uma tupla ``(remaining_params, indep_test_applied)``, onde:
      - ``remaining_params``: cópia de ``params`` SEM as chaves consumidas
        (``indep_test`` e ``alpha`` se ela acompanhava o teste).
      - ``indep_test_applied``: nome do teste efetivamente aplicado
        (``None`` se nada foi aplicado).

    Quando ``indep_test`` for ``"mv_fisherz"``, cai silenciosamente para
    Fisher-Z padrão (py-tetrad não tem o teste com tratamento de missing
    values; o equivalente seria pré-tratar os NaN antes — o que nossa
    pipeline já faz no ``_prepare_numeric_matrix``).
    """
    remaining = dict(params or {})
    test_name = remaining.pop("indep_test", None)
    if test_name is None:
        return remaining, None

    # mv_fisherz não tem equivalente no py-tetrad — usa Fisher-Z padrão.
    if test_name == "mv_fisherz":
        test_name = "fisherz"

    setter_name = _PYTETRAD_INDEP_TEST_SETTERS.get(test_name)
    if setter_name is None:
        # Teste desconhecido — não consome ``alpha``, devolve ``params`` como
        # está e deixa o caller cair no default do py-tetrad.
        _log(f"AVISO | indep_test desconhecido para py-tetrad: '{test_name}'.")
        remaining["indep_test"] = test_name  # devolve para o dict
        return remaining, None

    setter = getattr(search_obj, setter_name, None)
    if not callable(setter):
        _log(
            f"AVISO | py-tetrad não tem método '{setter_name}' "
            f"(esperado para indep_test='{test_name}'). Caindo no default."
        )
        return remaining, None

    # Alpha acompanha o teste como argumento nomeado. Consome do dict
    # para evitar que o loop genérico tente aplicar de novo via set_alpha.
    alpha = remaining.pop("alpha", None)
    try:
        if alpha is not None:
            setter(alpha=_coerce_param_value(alpha))
        else:
            setter()
        _log(f"py-tetrad: aplicou indep_test='{test_name}' via {setter_name}().")
        return remaining, test_name
    except Exception as exc:
        _log(
            f"AVISO | py-tetrad: falha ao chamar {setter_name}(alpha={alpha}): {exc}. "
            f"Devolvendo alpha para o loop genérico de setters."
        )
        if alpha is not None:
            remaining["alpha"] = alpha
        return remaining, None


# ----------------------------------------------------------------------
# Dispatcher causal-learn (fallback)
# ----------------------------------------------------------------------

def _run_with_causallearn(
    algorithm_key: str,
    data_matrix: np.ndarray,
    params: dict,
    col_names: list,
) -> dict:
    """Executa o algoritmo via causal-learn (puro Python, sem JVM).

    Suporta apenas pc, ges/fges e fci. Para algoritmos py-tetrad-only
    (boss, grasp, gfci, bfci, lv_lite), levanta erro claro.
    """
    if algorithm_key == "pc":
        result = _run_pc_with_causallearn(data_matrix, params)
        edges = _edges_to_tetrad_strings(result, col_names)
    elif algorithm_key in {"ges", "fges"}:
        result = _run_ges_with_causallearn(data_matrix)
        edges = _edges_to_tetrad_strings(result, col_names)
    elif algorithm_key == "fci":
        result = _run_fci_with_causallearn(data_matrix, params)
        edges = [str(e) for e in result.get("edges", [])]
        if not edges and "G" in result:
            edges = _edges_to_tetrad_strings(result, col_names)
    else:
        raise RuntimeError(
            f"Algorithm '{algorithm_key}' is not supported by the causal-learn fallback. "
            "Install/configure py-tetrad for this algorithm, or choose a different algorithm "
            "(pc, ges, fges, fci) that has causal-learn implementation."
        )
    return {"engine": "causal-learn", "result": result, "edges": edges}


# ----------------------------------------------------------------------
# Mapping algoritmo → backend declarado no registry
# ----------------------------------------------------------------------

# Espelho do campo ``backend`` em AlgorithmSpec (parameter_registry). Mantemos
# uma cópia local para evitar import circular e para que o service possa
# decidir a estratégia sem precisar carregar o registry inteiro.
ALGORITHM_BACKENDS = {
    "pc": "pytetrad_or_causallearn",
    "fges": "pytetrad_or_causallearn",
    "ges": "causallearn",
    "fci": "causallearn",
    "gfci": "placeholder",
    "boss": "pytetrad",
    "grasp": "pytetrad",
    "bfci": "pytetrad",
    "lv_lite": "pytetrad",
}


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def run_discovery(
    config: DiscoveryRunConfig,
    progress_callback: ProgressCallback = None,
) -> dict:
    """Executa um algoritmo de descoberta causal e gera os artefatos de saída.

    Pipeline:
      1. Carrega o dataset.
      2. Prepara matriz numérica (com fill diferenciado para pathways).
      3. Decide a estratégia conforme ``ALGORITHM_BACKENDS[algorithm_key]``:
           - "pytetrad_or_causallearn": tenta py-tetrad; cai para causal-learn.
           - "causallearn": vai direto para causal-learn (não tenta py-tetrad
             desnecessariamente).
           - "pytetrad": apenas py-tetrad; se falhar, propaga o erro.
           - "placeholder": levanta erro pedindo configuração específica.
      4. Escreve arestas (TETRAD), DOT (preservando tipos de aresta) e PNG.

    Parâmetros
    ----------
    config : DiscoveryRunConfig
        Configuração da execução (dataset, algoritmo, params, paths).
    progress_callback : callable, opcional
        Função ``(mensagem, progresso)`` para atualizações de UI.
    """
    _emit(progress_callback, f"Iniciando descoberta causal: algoritmo='{config.algorithm_key}'.", 0.02)

    _emit(progress_callback, f"Carregando dataset de {config.dataset_path.name}.", 0.10)
    df = _ensure_dataframe(config.dataset_path)
    _log(f"Dataset shape após leitura: {df.shape}")

    _emit(progress_callback, "Preparando matriz numérica (pathways → 0, clínicas → mediana).", 0.20)
    data_matrix, col_names, prep_debug = _prepare_numeric_matrix(df, progress_callback=progress_callback)

    out_dir = config.paths.discovery_dir / config.algorithm_key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "columns_used.txt").write_text("\n".join(col_names), encoding="utf-8")

    # DataFrame preparado para uso pelo py-tetrad (matriz numérica + nomes
    # das colunas). Esse é o objeto que **deve** ser passado para
    # TetradSearch — não o df cru.
    prepared_df = pd.DataFrame(data_matrix, columns=col_names)

    backend_strategy = ALGORITHM_BACKENDS.get(config.algorithm_key, "pytetrad_or_causallearn")
    _log(f"Estratégia de backend para '{config.algorithm_key}': {backend_strategy}")

    if backend_strategy == "placeholder":
        raise RuntimeError(
            f"Algorithm '{config.algorithm_key}' is marked as 'placeholder' in the registry. "
            "It needs a specific py-tetrad bridge implementation in the target environment. "
            "Choose another algorithm (pc, fges, ges, fci) for now."
        )

    run_debug: dict = {
        "algorithm_key": config.algorithm_key,
        "backend_strategy": backend_strategy,
        "py_tetrad_attempted": False,
        "py_tetrad_error": None,
        "fallback_used": False,
    }

    edges: list = []
    dot: Optional[str] = None
    engine_used: str = ""

    # Decisão da estratégia.
    try_pytetrad = backend_strategy in {"pytetrad_or_causallearn", "pytetrad"}
    allow_fallback = backend_strategy == "pytetrad_or_causallearn"

    if try_pytetrad:
        run_debug["py_tetrad_attempted"] = True
        _emit(progress_callback, "Tentando executar via py-tetrad.", 0.35)
        try:
            py_result = _attempt_pytetrad(config.algorithm_key, prepared_df, config.params, col_names)
            engine_used = py_result["engine"]
            edges = py_result["edges"]
            dot = py_result.get("dot")
            run_debug["engine_used"] = engine_used
            run_debug["py_tetrad"] = {
                "search_obj_type": py_result.get("search_obj_type"),
                "applied_params": py_result.get("applied_params", {}),
                **py_result.get("debug", {}),
            }
            _emit(progress_callback, f"py-tetrad concluído — {len(edges)} aresta(s) extraída(s).", 0.65)
        except Exception as exc:
            run_debug["py_tetrad_error"] = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _log(f"py-tetrad failed. Reason: {exc}")

            if allow_fallback:
                _emit(
                    progress_callback,
                    f"py-tetrad falhou ({type(exc).__name__}); usando causal-learn.",
                    0.45,
                )
                run_debug["fallback_used"] = True
                cl_result = _run_with_causallearn(
                    config.algorithm_key, data_matrix, config.params, col_names,
                )
                engine_used = cl_result["engine"]
                edges = cl_result["edges"]
                run_debug["engine_used"] = engine_used
                _emit(progress_callback, f"causal-learn concluído — {len(edges)} aresta(s) extraída(s).", 0.65)
            else:
                # Backend "pytetrad" puro: propaga o erro sem fallback.
                raise
    else:
        # backend_strategy == "causallearn"
        _emit(progress_callback, "Executando via causal-learn (estratégia direta do registry).", 0.35)
        cl_result = _run_with_causallearn(
            config.algorithm_key, data_matrix, config.params, col_names,
        )
        engine_used = cl_result["engine"]
        edges = cl_result["edges"]
        run_debug["engine_used"] = engine_used
        _emit(progress_callback, f"causal-learn concluído — {len(edges)} aresta(s) extraída(s).", 0.65)

    # ------------------------------------------------------------------
    # Persistência dos artefatos.
    # ------------------------------------------------------------------
    _emit(progress_callback, "Gerando arquivos de saída (edges, DOT, PNG).", 0.80)

    edges_path = out_dir / f"{config.algorithm_key}_edges.txt"
    edges_path.write_text("\n".join(edges), encoding="utf-8")

    parsed_edges = _parse_tetrad_edges(edges)

    # DOT — duas estratégias:
    #   1. Se o backend nos deu um DOT, usamos.
    #   2. Senão, geramos a partir do parse, preservando tipos de aresta
    #      (incluindo ---, o->, o-o que antes eram descartados).
    if dot is None:
        dot = _dot_from_tetrad_edges(parsed_edges)

    dot_path = out_dir / f"{config.algorithm_key}.dot"
    dot_path.write_text(dot, encoding="utf-8")

    image_path = out_dir / f"{config.algorithm_key}.png"
    image_rendered = False
    if parsed_edges:
        try:
            # save_dot_string_as_png aceita o DOT como string. Mantém
            # compatibilidade com o caller original.
            save_dot_string_as_png(dot, image_path)
            image_rendered = True
        except Exception as exc:
            _log(f"Falha ao renderizar PNG: {exc}")
            run_debug["png_render_error"] = str(exc)

    # Compatibilidade: lista de tuplas direcionadas (alguns consumers podem
    # ainda usar para análise downstream).
    directed_edges = _directed_edges_from_tetrad_strings(edges)

    (out_dir / "prep_debug.json").write_text(
        json.dumps(prep_debug, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "run_debug.json").write_text(
        json.dumps(run_debug, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(config.params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    _emit(
        progress_callback,
        f"Descoberta concluída: {len(edges)} aresta(s), engine={engine_used}.",
        1.0,
    )

    return {
        "output_dir": out_dir,
        "dataset_path": config.dataset_path,
        "edges_path": edges_path,
        "dot_path": dot_path,
        "image_path": image_path if image_rendered else None,
        "columns": col_names,
        "edges": edges,
        "parsed_edges": parsed_edges,
        "directed_edges": directed_edges,
        "engine_used": engine_used,
        "run_debug_path": out_dir / "run_debug.json",
    }