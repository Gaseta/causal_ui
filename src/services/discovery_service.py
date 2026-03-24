from __future__ import annotations

import importlib
import inspect
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.domain.models import DiscoveryRunConfig
from src.services.graph_utils import dot_from_edges, save_dot_string_as_png


def _log(msg: str) -> None:
    print(f"[discovery] {msg}", flush=True)


def _ensure_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False, index_col=0)
        except Exception:
            return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _prepare_numeric_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    debug: dict[str, Any] = {}
    out = df.copy()
    dropped = [c for c in ["PATIENT_ID", "SAMPLE_ID"] if c in out.columns]
    if dropped:
        out = out.drop(columns=dropped)
    debug["dropped_id_columns"] = dropped

    categorical_cols: list[str] = []
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            converted = pd.to_numeric(out[c], errors="coerce")
            frac_numeric = float(converted.notna().mean())
            if frac_numeric >= 0.95:
                out[c] = converted
            else:
                categorical_cols.append(c)

    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder

        for c in categorical_cols:
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype(str))

    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.fillna(out.median(numeric_only=True)).fillna(0.0).astype(float)

    zero_var = list(out.columns[out.std(axis=0, ddof=0) == 0.0])
    if zero_var:
        rng = np.random.default_rng(42)
        for c in zero_var:
            out[c] = out[c] + rng.normal(0.0, 1e-6, size=len(out))

    debug["final_shape"] = out.shape
    debug["categorical_encoded"] = categorical_cols
    debug["final_columns"] = out.columns.tolist()
    return out.to_numpy(dtype=float), out.columns.tolist(), debug


def _edges_to_tetrad_strings(result: Any, variable_names: List[str]) -> List[str]:
    from causallearn.graph.Endpoint import Endpoint

    G = result.G if hasattr(result, "G") else result["G"]

    def resolve_name(node_name: str) -> str:
        s = str(node_name)
        m = re.search(r"(\d+)$", s)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(variable_names):
                return variable_names[idx]
            if 1 <= idx <= len(variable_names):
                return variable_names[idx - 1]
        return s

    edges: List[str] = []
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


def _directed_edges_from_tetrad_strings(edges: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pat = re.compile(r"^(.*?)\s(-->|<--)\s(.*?)$")
    for edge in edges:
        m = pat.match(edge.strip())
        if not m:
            continue
        a, rel, b = m.group(1).strip(), m.group(2), m.group(3).strip()
        out.append((a, b) if rel == "-->" else (b, a))
    seen = set()
    uniq = []
    for item in out:
        if item not in seen and item[0] != item[1]:
            seen.add(item)
            uniq.append(item)
    return uniq


def _run_pc_with_causallearn(data_matrix: np.ndarray, params: dict[str, Any]):
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


def _run_fci_with_causallearn(data_matrix: np.ndarray, params: dict[str, Any]):
    from causallearn.search.ConstraintBased.FCI import fci

    graph, edges = fci(
        data_matrix,
        independence_test_method=params.get("indep_test", "fisherz"),
        alpha=float(params.get("alpha", 0.05)),
    )
    return {"G": graph, "edges": edges}


def _load_tetrad_search_class():
    candidates = [
        ("tools.TetradSearch", "TetradSearch"),
        ("pytetrad.tools.TetradSearch", "TetradSearch"),
    ]
    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        except Exception as e:
            errors.append(f"{module_name}: {e}")
    raise RuntimeError("Could not import TetradSearch. " + " | ".join(errors))


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _apply_params_to_object(obj: Any, params: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
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
            if callable(method):
                try:
                    method(value)
                    applied[key] = value
                    break
                except TypeError:
                    try:
                        method(**{key: value})
                        applied[key] = value
                        break
                    except Exception:
                        continue
                except Exception:
                    continue
    return applied


def _invoke_algorithm(search_obj: Any, algorithm_key: str, params: dict[str, Any]) -> Any:
    candidates = {
        "pc": ["run_pc", "pc"],
        "fci": ["run_fci", "fci"],
        "ges": ["run_ges", "ges", "run_fges", "fges"],
        "fges": ["run_fges", "fges", "run_ges", "ges"],
        "boss": ["run_boss", "boss"],
        "grasp": ["run_grasp", "grasp"],
        "bfci": ["run_bfci", "bfci"],
        "lv_lite": ["run_lv_lite", "lv_lite", "run_lvlite", "lvlite"],
    }
    names = candidates.get(algorithm_key, [f"run_{algorithm_key}", algorithm_key])
    last_error: Exception | None = None

    for name in names:
        method = getattr(search_obj, name, None)
        if not callable(method):
            continue
        try:
            sig = inspect.signature(method)
            accepted_kwargs = {
                k: _coerce_param_value(v)
                for k, v in params.items()
                if k in sig.parameters
            }
            if accepted_kwargs:
                return method(**accepted_kwargs)
            if len(sig.parameters) == 0:
                return method()
            return method(*[])
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise last_error
    raise AttributeError(f"No callable py-tetrad method found for algorithm '{algorithm_key}'.")


def _extract_edges_from_string(text: str) -> list[str]:
    rel_tokens = ("-->", "<--", "---", "o->", "<-o", "o-o")
    edges: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s and any(tok in s for tok in rel_tokens):
            edges.append(s)
    return edges


def _extract_pytetrad_result(search_obj: Any, result: Any, variable_names: list[str]) -> tuple[list[str], str | None, dict[str, Any]]:
    debug: dict[str, Any] = {"result_type": type(result).__name__}
    dot: str | None = None

    if isinstance(result, dict) and "edges" in result:
        edges = [str(e) for e in result["edges"]]
        return edges, result.get("dot"), debug

    if isinstance(result, list):
        return [str(e) for e in result], dot, debug

    if isinstance(result, str):
        edges = _extract_edges_from_string(result)
        if edges:
            return edges, dot, debug

    if hasattr(result, "get_graph_edges") or hasattr(result, "G"):
        try:
            edges = _edges_to_tetrad_strings(result, variable_names)
            return edges, dot, debug
        except Exception as e:
            debug["graph_parse_error"] = str(e)

    for attr_name in ["get_string", "get_graph_as_string", "get_string_graph", "get_result_graph_string"]:
        fn = getattr(search_obj, attr_name, None)
        if callable(fn):
            try:
                text = fn()
                if text is not None:
                    edges = _extract_edges_from_string(str(text))
                    if edges:
                        debug["edges_source"] = attr_name
                        break
            except Exception as e:
                debug[f"{attr_name}_error"] = str(e)
    else:
        edges = []

    for attr_name in ["get_dot", "get_dot_graph", "get_dot_string"]:
        fn = getattr(search_obj, attr_name, None)
        if callable(fn):
            try:
                maybe_dot = fn()
                if maybe_dot:
                    dot = str(maybe_dot)
                    debug["dot_source"] = attr_name
                    break
            except Exception as e:
                debug[f"{attr_name}_error"] = str(e)

    if not edges:
        raise RuntimeError("py-tetrad executed but no edges could be extracted from the result.")
    return edges, dot, debug


def _attempt_pytetrad(algorithm_key: str, df: pd.DataFrame, params: dict[str, Any], variable_names: list[str]):
    TetradSearch = _load_tetrad_search_class()
    search_obj = TetradSearch(df.copy())
    applied_params = _apply_params_to_object(search_obj, params)
    result = _invoke_algorithm(search_obj, algorithm_key, params)
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


def _run_with_causallearn(algorithm_key: str, data_matrix: np.ndarray, params: dict[str, Any], col_names: list[str]):
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
            "Install/configure py-tetrad for this algorithm."
        )
    return {"engine": "causal-learn", "result": result, "edges": edges}


def run_discovery(config: DiscoveryRunConfig) -> dict[str, Any]:
    df = _ensure_dataframe(config.dataset_path)
    data_matrix, col_names, debug = _prepare_numeric_matrix(df)
    out_dir = config.paths.discovery_dir / config.algorithm_key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "columns_used.txt").write_text("\n".join(col_names), encoding="utf-8")

    run_debug: dict[str, Any] = {
        "requested_engine_order": ["py-tetrad", "causal-learn"],
        "algorithm_key": config.algorithm_key,
        "py_tetrad_error": None,
        "fallback_used": False,
    }

    dot: str | None = None
    try:
        _log(f"Trying py-tetrad first for algorithm '{config.algorithm_key}'")
        py_result = _attempt_pytetrad(config.algorithm_key, df, config.params, col_names)
        engine_used = py_result["engine"]
        edges = py_result["edges"]
        dot = py_result.get("dot")
        run_debug["engine_used"] = engine_used
        run_debug["py_tetrad"] = {
            "search_obj_type": py_result.get("search_obj_type"),
            "applied_params": py_result.get("applied_params", {}),
            **py_result.get("debug", {}),
        }
    except Exception as e:
        run_debug["py_tetrad_error"] = {
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        run_debug["fallback_used"] = True
        _log(f"py-tetrad failed; falling back to causal-learn. Reason: {e}")
        cl_result = _run_with_causallearn(config.algorithm_key, data_matrix, config.params, col_names)
        engine_used = cl_result["engine"]
        edges = cl_result["edges"]
        run_debug["engine_used"] = engine_used

    edges_path = out_dir / f"{config.algorithm_key}_edges.txt"
    edges_path.write_text("\n".join(edges), encoding="utf-8")

    directed_edges = _directed_edges_from_tetrad_strings(edges)
    if dot is None:
        dot = dot_from_edges(directed_edges)

    dot_path = out_dir / f"{config.algorithm_key}.dot"
    dot_path.write_text(dot, encoding="utf-8")

    image_path = out_dir / f"{config.algorithm_key}.png"
    if directed_edges:
        save_dot_string_as_png(dot, image_path)

    (out_dir / "prep_debug.json").write_text(
        json.dumps(debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "run_debug.json").write_text(
        json.dumps(run_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(config.params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_dir": out_dir,
        "dataset_path": config.dataset_path,
        "edges_path": edges_path,
        "dot_path": dot_path,
        "image_path": image_path if image_path.exists() else None,
        "columns": col_names,
        "edges": edges,
        "engine_used": engine_used,
        "run_debug_path": out_dir / "run_debug.json",
    }
