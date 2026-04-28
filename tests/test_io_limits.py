from __future__ import annotations

import pandas as pd
import pytest

import src.io_load as io_load
import src.session_io as session_io


def test_load_uploaded_any_rejects_large_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io_load, "MAX_UPLOAD_BYTES", 4)

    with pytest.raises(ValueError, match="too large"):
        io_load.load_uploaded_any(b"src,dst\n1,2\n", "edges.csv")


def test_validate_table_size_rejects_too_many_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(io_load, "MAX_UPLOAD_COLUMNS", 2)

    with pytest.raises(ValueError, match="too many columns"):
        io_load.validate_table_size(pd.DataFrame([[1, 2, 3]]), label="Upload")


def test_import_workspace_rejects_large_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_io, "MAX_WORKSPACE_JSON_BYTES", 2)

    with pytest.raises(ValueError, match="Workspace JSON is too large"):
        session_io.import_workspace_json(b'{"graphs": {}, "experiments": []}')
