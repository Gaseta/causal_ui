from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd


def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "unnamed"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(r"^#+\s*", "", regex=True)
    )
    return df


def detect_separator(path: os.PathLike | str, preferred_sep: str | None = None) -> str:
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(16384)

    if preferred_sep and preferred_sep in sample:
        return preferred_sep

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return dialect.delimiter
    except Exception:
        pass

    candidates = {
        "\t": sample.count("\t"),
        ",": sample.count(","),
        ";": sample.count(";"),
        "|": sample.count("|"),
    }

    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else ","


def read_table(
    path: os.PathLike | str,
    preferred_sep: str | None = "\t",
    usecols=None,
    skiprows: int = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    path = os.fspath(path)

    def _read(sep, engine=None):
        return pd.read_csv(
            path,
            sep=sep,
            engine=engine,
            comment="#",
            low_memory=False,
            dtype=str,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
        )

    detected_sep = detect_separator(path, preferred_sep=preferred_sep)

    read_attempts = [
        (detected_sep, None),
        ("\t", None),
        (",", None),
        (";", None),
        ("|", None),
        (None, "python"),
    ]

    best_df: pd.DataFrame | None = None

    for sep, engine in read_attempts:
        try:
            df = normalize_columns(_read(sep=sep, engine=engine))
        except Exception:
            continue

        if best_df is None or df.shape[1] > best_df.shape[1]:
            best_df = df

        if df.shape[1] > 1:
            return df

    if best_df is None:
        raise ValueError(f"Não foi possível ler a tabela: {path}")

    return best_df


def inspect_table(path: os.PathLike | str, max_rows: int = 5) -> tuple[pd.DataFrame, list[str]]:
    df = read_table(path, nrows=max_rows)
    return df.head(max_rows), list(df.columns)


def write_uploaded_file(uploaded_file: BinaryIO, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.read()
    destination.write_bytes(data)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return destination


def write_uploaded_bytes(data: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}. Columns available: {list(df.columns)[:120]}")


def temp_workdir(prefix: str = "causal_ui_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))