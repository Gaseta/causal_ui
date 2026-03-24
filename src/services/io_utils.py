from __future__ import annotations

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


def read_table(path: os.PathLike | str, preferred_sep: str = "\t", usecols=None, skiprows: int = 0) -> pd.DataFrame:
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
        )

    df = normalize_columns(_read(preferred_sep))
    if df.shape[1] <= 1:
        try:
            df2 = normalize_columns(_read(sep=None, engine="python"))
            if df2.shape[1] > df.shape[1]:
                df = df2
        except Exception:
            pass
    if df.shape[1] <= 1:
        try:
            df3 = normalize_columns(_read(sep=","))
            if df3.shape[1] > df.shape[1]:
                df = df3
        except Exception:
            pass
    return df


def inspect_table(path: os.PathLike | str, max_rows: int = 5) -> tuple[pd.DataFrame, list[str]]:
    df = read_table(path)
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
