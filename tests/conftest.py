"""Shared test helpers. The test data in tests/data is made up (see make_test_data.py)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

DATA = Path(__file__).parent / "data"


def write_rds(csv: Path, rds: Path) -> Path:
    """Save a test count table as .rds (data.frame), as R would."""
    pyreadr = pytest.importorskip("pyreadr")
    pyreadr.write_rds(str(rds), pl.read_csv(csv).to_pandas())
    return rds
