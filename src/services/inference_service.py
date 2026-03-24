from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dowhy import CausalModel

from src.domain.models import InferenceRunConfig


_DOT_EDGE_RE = re.compile(r'^\s*"(.*?)"\s*->\s*"(.*?)"\s*;\s*$')
_EDGE_RE = re.compile(r'^(.*?)\s(-->|<--|---|o->|<-o|o-o)\s(.*?)$')


def _log(msg: str) -> None:
    print(f"[inference] {msg}", flush=True)


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False, index_col=0)
        except Exception:
            return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _graph_to_dot(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "digraph" in text:
        return text
    directed: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EDGE_RE.match(line)
        if not m:
            continue
        a, rel, b = m.group(1).strip(), m.group(2), m.group(3).strip()
        if rel == "-->":
            directed.append((a, b))
        elif rel == "<--":
            directed.append((b, a))
    lines = ["digraph G {"]
    for s, t in directed:
        lines.append(f'  "{s}" -> "{t}";')
    lines.append("}")
    return "\n".join(lines)


def _vars_from_dot(dot: str) -> list[str]:
    out = set()
    for ln in dot.splitlines():
        m = _DOT_EDGE_RE.match(ln)
        if m:
            out.add(m.group(1))
            out.add(m.group(2))
    return sorted(out)


def _extract_backdoor_vars(estimand: Any) -> list[str]:
    for attr in ["backdoor_variables", "backdoor_vars", "backdoor_set"]:
        if hasattr(estimand, attr):
            value = getattr(estimand, attr)
            if isinstance(value, (list, tuple, set)):
                return [str(v) for v in value]
    s = str(estimand)
    m = re.search(r"Backdoor\s+variables\s*:\s*\[(.*?)\]", s, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return [p.strip().strip("'\"") for p in m.group(1).split(",") if p.strip()]


def _coerce_model_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[[c for c in cols if c in df.columns]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna()


def _pair_iterator(treatments: list[str], outcomes: list[str], all_pairs: bool):
    if all_pairs:
        for t in treatments:
            for o in outcomes:
                if t != o:
                    yield t, o
    else:
        for t in treatments:
            for o in outcomes:
                if t != o:
                    yield t, o


def run_inference(config: InferenceRunConfig) -> dict[str, Any]:
    df = _read_dataset(config.dataset_path)
    dot = _graph_to_dot(config.graph_path)
    graph_vars = _vars_from_dot(dot)

    out_dir = config.paths.causal_results_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_used.dot").write_text(dot, encoding="utf-8")

    summary: dict[str, Any] = {"pairs": [], "result_dir": str(out_dir)}

    for treatment, outcome in _pair_iterator(config.treatment_columns, config.outcome_columns, config.all_pairs):
        needed = sorted(set(graph_vars + [treatment, outcome]))
        model_df = _coerce_model_df(df, needed)
        if len(model_df) < 20 or treatment not in model_df.columns or outcome not in model_df.columns:
            continue

        model = CausalModel(data=model_df, treatment=treatment, outcome=outcome, graph=dot)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name=config.estimate_method, method_params=config.estimate_params)

        pair_slug = re.sub(r'[^a-zA-Z0-9_.-]+', '_', f"T={treatment}__Y={outcome}")
        pair_dir = out_dir / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)

        estimate_payload = {
            "treatment": treatment,
            "outcome": outcome,
            "estimate_method": config.estimate_method,
            "estimate_params": config.estimate_params,
            "backdoor_vars": _extract_backdoor_vars(estimand),
        }
        (pair_dir / "estimate_metadata.json").write_text(json.dumps(estimate_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (pair_dir / "identified_estimand.txt").write_text(str(estimand), encoding="utf-8")
        (pair_dir / "estimate.txt").write_text(str(estimate), encoding="utf-8")

        refuter_outputs: dict[str, str] = {}
        for refuter in config.refuters:
            try:
                ref = model.refute_estimate(
                    estimand,
                    estimate,
                    method_name=refuter,
                    **config.refuter_params.get(refuter, {}),
                )
                refuter_outputs[refuter] = str(ref)
            except Exception as e:
                refuter_outputs[refuter] = f"{type(e).__name__}: {e}"
            (pair_dir / f"refuter__{refuter}.txt").write_text(refuter_outputs[refuter], encoding="utf-8")

        summary["pairs"].append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "pair_dir": str(pair_dir),
                "estimate_method": config.estimate_method,
                "refuters": list(refuter_outputs.keys()),
            }
        )
        _log(f"Completed pair treatment={treatment} outcome={outcome}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
