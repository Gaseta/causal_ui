from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx
import pydot


def save_nx_as_png(G: nx.Graph, out_path: str | Path, *, prog: str = "dot", dpi: int = 180) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pydot_graph = nx.nx_pydot.to_pydot(G)
    pydot_graph.set_rankdir("LR")
    pydot_graph.set_dpi(str(dpi))
    png_bytes = pydot_graph.create_png(prog=prog)
    out_path.write_bytes(png_bytes)
    return out_path


def save_dot_string_as_png(dot_str: str, out_path: str | Path, *, prog: str = "dot", dpi: int = 180) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graphs = pydot.graph_from_dot_data(dot_str)
    if not graphs:
        raise ValueError("DOT inválido: não foi possível parsear.")
    g = graphs[0]
    g.set_rankdir("LR")
    g.set_dpi(str(dpi))
    png_bytes = g.create_png(prog=prog)
    out_path.write_bytes(png_bytes)
    return out_path


def dot_from_edges(edges: Iterable[tuple[str, str]]) -> str:
    lines = ["digraph G {"]
    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)
