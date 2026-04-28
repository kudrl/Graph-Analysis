from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

from .preprocess import coerce_fixed_format

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 300_000
MAX_UPLOAD_COLUMNS = 100


def load_edges(path_or_bytes: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    if isinstance(path_or_bytes, (bytes, bytearray)):
        use_name = filename or ""
        return load_uploaded_any(bytes(path_or_bytes), use_name)
    raise TypeError("path_or_bytes must be a file path or raw bytes")


def load_uploaded_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded file is too large: max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    else:
        try:
            df = pd.read_csv(bio, sep=None, engine="python", encoding_errors="replace")
        except (UnicodeDecodeError, ParserError):
            bio.seek(0)
            df = pd.read_csv(
                bio,
                sep=None,
                engine="python",
                encoding="cp1251",
            )

    df.columns = [str(c).strip() for c in df.columns]
    validate_table_size(df, label="Uploaded file")
    return df


def validate_table_size(df: pd.DataFrame, *, label: str = "Table") -> None:
    rows, cols = df.shape
    if rows > MAX_UPLOAD_ROWS:
        raise ValueError(f"{label} has too many rows: {rows:,} > {MAX_UPLOAD_ROWS:,}.")
    if cols > MAX_UPLOAD_COLUMNS:
        raise ValueError(f"{label} has too many columns: {cols:,} > {MAX_UPLOAD_COLUMNS:,}.")


def clean_fixed_format(df_any: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, meta0 = coerce_fixed_format(df_any)
    meta = {
        "SRC_COL": meta0["src_col"],
        "DST_COL": meta0["dst_col"],
        "CONF_COL": "confidence",
        "WEIGHT_COL": "weight",
    }
    return df, meta
