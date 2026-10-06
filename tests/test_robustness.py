"""Unusual and broken inputs found by stress testing: correct result or a clear error."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from trace_rna import compute_siteindex
from trace_rna.cli import main

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def tables(tmp_path):
    t = pl.read_csv(DATA / "ai_treated.csv")
    c = pl.read_csv(DATA / "ai_control.csv")
    t.write_csv(tmp_path / "t.csv")
    c.write_csv(tmp_path / "c.csv")
    return tmp_path, t, c


def _run(tmp, *extra, method="AI", treated="t.csv", control="c.csv", out="o.csv"):
    return main(["-q", "siteindex", "--method", method, "--treated", str(tmp / treated),
                 "--control", str(tmp / control), "-o", str(tmp / out), *extra])


def _result(tmp, name="o.csv"):
    return pl.read_csv(tmp / name, null_values="NA").drop("datafile1", "datafile2")


def test_output_never_overwrites_an_input(tables, caplog):
    tmp, _, _ = tables
    before = (tmp / "t.csv").read_bytes()
    assert _run(tmp, "--overwrite", out="t.csv") == 1
    assert (tmp / "t.csv").read_bytes() == before
    assert "also an input file" in caplog.text


@pytest.mark.parametrize("variant", ["bom", "crlf", "quoted", "lower_refseq", "float_gencoor", "gz"])
def test_excel_and_format_variants_give_same_result(tables, variant):
    tmp, t, _ = tables
    assert _run(tmp, out="ref.csv") == 0
    text = (tmp / "t.csv").read_text()
    name = f"t_{variant}.csv"
    if variant == "bom":
        (tmp / name).write_bytes(("﻿" + text).encode())
    elif variant == "crlf":
        (tmp / name).write_bytes(text.replace("\n", "\r\n").encode())
    elif variant == "quoted":
        (tmp / name).write_text("\n".join(",".join(f'"{v}"' for v in ln.split(",")) for ln in text.splitlines()) + "\n")
    elif variant == "lower_refseq":
        t.with_columns(pl.col("refSeq").str.to_lowercase()).write_csv(tmp / name)
    elif variant == "float_gencoor":
        t.with_columns(pl.col("gencoor").cast(pl.Float64)).write_csv(tmp / name)
    elif variant == "gz":
        name = "t.csv.gz"
        with gzip.open(tmp / name, "wb") as fh:
            fh.write(text.encode())
    assert _run(tmp, treated=name) == 0
    assert _result(tmp).equals(_result(tmp, "ref.csv"))


def test_u_in_refseq_is_read_as_t():
    x = pl.DataFrame({"chr": ["a"], "gencoor": [1], "cov": [50], "refSeq": ["u"],
                      "A": [0], "C": [0], "G": [0], "T": [40], "-": [10]})
    assert compute_siteindex(x, x, "pseudo", cov_threshold=0)["siteindex.x"].item() == pytest.approx(0.2)


def test_method_names_ignore_case(tables):
    tmp, _, _ = tables
    assert _run(tmp, method="ai") == 0
    assert _run(tmp, method="aI", out="o2.csv") == 0


@pytest.mark.parametrize("case, message", [
    ("semicolon", "uses ';' as separator"),
    ("text_in_counts", "could not parse"),
    ("corrupt_gz", "File problem"),
    ("empty", "Could not read the input"),
])
def test_clear_errors_instead_of_crashes(tables, caplog, case, message):
    tmp, t, _ = tables
    name = f"bad_{case}.csv"
    if case == "semicolon":
        (tmp / name).write_text((tmp / "t.csv").read_text().replace(",", ";"))
    elif case == "text_in_counts":
        t.with_columns(pl.when(pl.int_range(pl.len()) == 3).then(pl.lit("abc"))
                       .otherwise(pl.col("A").cast(pl.String)).alias("A")).write_csv(tmp / name)
    elif case == "corrupt_gz":
        name = "bad.csv.gz"
        (tmp / name).write_bytes(b"\x1f\x8b broken")
    elif case == "empty":
        (tmp / name).write_text("")
    assert _run(tmp, treated=name) == 1
    assert message in caplog.text


def test_german_excel_sample_sheet(tables, caplog):
    tmp, _, _ = tables
    (tmp / "sheet.csv").write_text("﻿treated;control;method\nt.csv;c.csv;AI\n")
    assert main(["-q", "batch", str(tmp / "sheet.csv"), "--outdir", str(tmp / "out")]) == 1
    assert "uses ';' as separator" in caplog.text
