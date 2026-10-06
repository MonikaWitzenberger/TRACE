from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pyreadr

from conftest import write_rds

from trace_rna.cli import main
from trace_rna.rds import ensure_parquet, parquet_path_for

DATA = Path(__file__).parent / "data"


def _copy(tmp_path: Path) -> Path:
    return write_rds(DATA / "ai_treated.csv", tmp_path / "sample.rds")


def test_converts_once_then_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_RSCRIPT", str(tmp_path / "no_R_here"))  # force the Python reader
    rds = _copy(tmp_path)
    out = ensure_parquet(rds)
    assert out == tmp_path / "sample.parquet"
    original = pyreadr.read_r(str(rds))[None]
    assert pl.read_parquet(out).height == len(original)
    stamp = out.stat().st_mtime
    assert ensure_parquet(rds) == out and out.stat().st_mtime == stamp  # reused, not rewritten


def test_reconverts_when_rds_is_newer(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_RSCRIPT", str(tmp_path / "no_R_here"))
    rds = _copy(tmp_path)
    out = ensure_parquet(rds)
    old = out.stat().st_mtime - 100
    os.utime(out, (old, old))  # parquet now older than the rds
    ensure_parquet(rds)
    assert out.stat().st_mtime > old


def test_unwritable_folder_uses_cache(tmp_path, monkeypatch):
    rds = _copy(tmp_path)
    monkeypatch.setattr(os, "access", lambda *a, **k: False)
    assert parquet_path_for(rds).parent == Path.home() / ".cache" / "trace_rna"


def test_scoring_directly_from_rds(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_RSCRIPT", str(tmp_path / "no_R_here"))
    t, c = tmp_path / "t.rds", tmp_path / "c.rds"
    write_rds(DATA / "ai_treated.csv", t)
    write_rds(DATA / "ai_control.csv", c)
    out = tmp_path / "o.csv"
    assert main(["-q", "siteindex", "--treated", str(t), "--control", str(c), "--method", "AI", "-o", str(out)]) == 0
    assert (tmp_path / "t.parquet").exists() and (tmp_path / "c.parquet").exists()
    first = out.read_text()
    assert main(["-q", "siteindex", "--treated", str(t), "--control", str(c), "--method", "AI",
                 "-o", str(out), "--overwrite"]) == 0
    assert out.read_text() == first


def test_rds_and_csv_give_same_results(tmp_path, monkeypatch):
    """Count tables as .rds, .csv and .csv.gz (same columns) give identical results."""
    import gzip

    from trace_rna import compute_geneindex, compute_siteindex

    monkeypatch.setenv("TRACE_RSCRIPT", str(tmp_path / "no_R_here"))
    for name in ("ai_treated", "ai_control"):
        write_rds(DATA / f"{name}.csv", tmp_path / f"{name}.rds")
        with gzip.open(tmp_path / f"{name}.csv.gz", "wb") as fh:
            fh.write((DATA / f"{name}.csv").read_bytes())
    files = ["datafile1", "datafile2"]  # differ by design (file names)
    ref_site = compute_siteindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "pseudo").drop(files)
    ref_gene = compute_geneindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI").drop(files)
    for ext in ("rds", "csv.gz"):
        t, c = tmp_path / f"ai_treated.{ext}", tmp_path / f"ai_control.{ext}"
        site = compute_siteindex(t, c, "pseudo")
        assert site["datafile1"][0] == f"ai_treated.{ext}"
        assert site.drop(files).equals(ref_site)
        assert compute_geneindex(t, c, "AI").drop(files).equals(ref_gene)


def test_large_text_tables_are_converted_once(tmp_path, monkeypatch):
    """Big CSV / CSV.gz count tables get a Parquet copy and give identical results."""
    import gzip

    import trace_rna.io as io_mod
    from trace_rna import compute_siteindex

    monkeypatch.setattr(io_mod, "TEXT_TO_PARQUET_BYTES", 0)  # treat every text table as large
    files = ["datafile1", "datafile2"]
    for name in ("ai_treated", "ai_control"):
        df = pl.read_csv(DATA / f"{name}.csv")
        with gzip.open(tmp_path / f"{name}.csv.gz", "wb") as fh:
            df.write_csv(fh)
        df.write_csv(tmp_path / f"{name}.tsv", separator="\t")
    ref = compute_siteindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI").drop(files)
    for ext in ("csv.gz", "tsv"):
        out = compute_siteindex(tmp_path / f"ai_treated.{ext}", tmp_path / f"ai_control.{ext}", "AI")
        assert out.drop(files).equals(ref)
        assert (tmp_path / "ai_treated.parquet").exists()
        (tmp_path / "ai_treated.parquet").unlink()
        (tmp_path / "ai_control.parquet").unlink()
    assert not list(tmp_path.glob(".*"))  # no temporary files left
