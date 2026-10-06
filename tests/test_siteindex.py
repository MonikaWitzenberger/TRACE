from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trace_rna import Method, Resources, compute_siteindex, register_method
from trace_rna.cli import main
from trace_rna.io import InputError, read_positions
from trace_rna.stats import chisq_2x2_pvalue

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------- statistics
def test_chisq_matches_scipy():
    """Same p-values as the standard 2x2 chi-square test (scipy, with continuity correction)."""
    from scipy.stats import chi2_contingency

    tables = [(24000, 6, 22000, 0), (900, 100, 950, 50), (10, 3, 12, 1), (5000, 5000, 5000, 5000)]
    p, na = chisq_2x2_pvalue(*(np.array(col, dtype=float) for col in zip(*tables)))
    for got, (a, b, c, d) in zip(p, tables):
        assert got == pytest.approx(chi2_contingency([[a, b], [c, d]], correction=True)[1], rel=1e-12)
    assert not na.any()


def test_chisq_na_and_nan_rules():
    p, na = chisq_2x2_pvalue(
        np.array([1.0, np.nan, 0.0]), np.array([1.0, 5, 0]),
        np.array([0.0, 5, 3]), np.array([1.0, 5, 5]),
    )
    assert na.tolist() == [True, True, False]  # total < 4; missing count
    assert np.isnan(p[2])  # zero row -> expected 0 -> NaN


# ---------------------------------------------------------------- helpers
def _sample(counts: dict[str, list], chr_="tx_1", start=1) -> pl.DataFrame:
    n = len(counts["refSeq"])
    base = {"chr": [chr_] * n, "gencoor": list(range(start, start + n))}
    cols = {"A": [0] * n, "C": [0] * n, "G": [0] * n, "T": [0] * n, "-": [0] * n}
    cols.update({k: v for k, v in counts.items() if k != "refSeq"})
    cov = [sum(cols[b][i] for b in "ACGT-") for i in range(n)]
    return pl.DataFrame({**base, "refSeq": counts["refSeq"], "cov": cov, **cols})


def test_label_columns(tmp_path):
    """experiment, datafile1 (treated) and datafile2 (control) are the last three columns."""
    out = compute_siteindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI", experiment="exp1")
    assert out.columns[-3:] == ["experiment", "datafile1", "datafile2"]
    assert out.select("experiment", "datafile1", "datafile2").unique().rows() == [
        ("exp1", "ai_treated.csv", "ai_control.csv")]
    in_memory = compute_siteindex(_sample({"refSeq": ["A"], "A": [50]}), _sample({"refSeq": ["A"], "A": [50]}),
                                  "AI", cov_threshold=0)
    assert in_memory.row(0, named=True)["datafile1"] is None  # no file name for in-memory tables


def test_positions_split_at_last_underscore(tmp_path):
    f = tmp_path / "snps.csv"
    f.write_text("pos\nmy_gene_1_12\ntx9_5\n")
    df = read_positions(f).collect().sort("chr")
    assert df.rows() == [("my_gene_1", 12), ("tx9", 5)]


def test_positions_bad_value(tmp_path):
    f = tmp_path / "snps.csv"
    f.write_text("pos\nnocoordinate\n")
    with pytest.raises(InputError, match="chr_position"):
        read_positions(f)


# ---------------------------------------------------------------- scoring
def test_basic_scores_and_filters():
    x = _sample({"refSeq": ["A", "A", "A", "C"], "A": [90, 10, 50, 0], "G": [10, 0, 50, 0], "C": [0, 0, 0, 100]})
    y = _sample({"refSeq": ["A", "A", "A", "C"], "A": [100, 10, 100, 0], "G": [0, 0, 0, 0], "C": [0, 0, 0, 100]})
    res = Resources(snps=pl.DataFrame({"chr": ["tx_1"], "gencoor": [3]}))
    out = compute_siteindex(x, y, "AI", cov_threshold=20, resources=res, experiment="exp1")
    # site 2: coverage 10 (<= 20); site 3: SNP; site 4: not A
    assert out["pos"].to_list() == ["tx_1_1"]
    row = out.row(0, named=True)
    assert row["siteindex.x"] == pytest.approx(0.1)
    assert row["siteindex.y"] == 0
    assert row["log2FC"] == float("inf")
    assert row["diff"] == pytest.approx(0.1)
    assert row["experiment"] == "exp1" and "target" not in out.columns
    assert out.schema["A.x"] == pl.Float64  # counts are doubles


def test_motif_uses_coordinates_not_rows():
    # Sequence C C C G: only position 3 is in a CCG motif (ac4C_CCG).
    # The treated sample lacks position 2, so the row before position 3 is position 1 ("C").
    # A row-order lookup would accept position 3; the true neighbour is unknown -> dropped.
    seq = {"refSeq": ["C", "C", "C", "G"], "C": [50, 50, 50, 0], "T": [50, 50, 50, 0], "G": [0, 0, 0, 100]}
    x = _sample(seq).filter(pl.col("gencoor") != 2)
    y = _sample(seq)
    assert compute_siteindex(x, y, "ac4C_CCG", cov_threshold=0)["pos"].to_list() == []
    assert compute_siteindex(y, y, "ac4C_CCG", cov_threshold=0)["pos"].to_list() == ["tx_1_3"]


def test_duplicate_positions_rejected():
    x = _sample({"refSeq": ["A", "A"], "A": [50, 50]}).with_columns(pl.lit(1).alias("gencoor"))
    with pytest.raises(ValueError, match="share a chr/gencoor"):
        compute_siteindex(x, x, "AI", cov_threshold=0)


def test_reference_required():
    x = _sample({"refSeq": ["T"], "T": [50]})
    with pytest.raises(ValueError, match="reference"):
        compute_siteindex(x, x, "m5C")


def test_unknown_method():
    x = _sample({"refSeq": ["T"], "T": [50]})
    with pytest.raises(ValueError, match="Available methods"):
        compute_siteindex(x, x, "m7G_typo")


def test_custom_method_registration():
    register_method(Method("test_GA", "test", "G", ("A",), ("G",)), replace=True)
    x = _sample({"refSeq": ["G"], "G": [75], "A": [25]})
    out = compute_siteindex(x, x, "test_GA", cov_threshold=0)
    assert out["siteindex.x"].item() == pytest.approx(0.25)


# ---------------------------------------------------------------- CLI
def test_cli_siteindex_and_overwrite(tmp_path):
    out = tmp_path / "res" / "ai.csv"
    args = ["-q", "siteindex", "--treated", str(DATA / "ai_treated.csv"),
            "--control", str(DATA / "ai_control.csv"), "--method", "AI",
            "--snps", str(DATA / "snps.csv"),
            "--experiment", "exp1", "-o", str(out)]
    assert main(args) == 0
    df = pl.read_csv(out, null_values="NA")
    assert df.columns[:2] == ["pos", "chr.x"] and df.columns[-3:] == ["experiment", "datafile1", "datafile2"]
    assert df["datafile1"].unique().to_list() == ["ai_treated.csv"]
    assert main(args) == 1  # refuses to overwrite
    assert main([*args, "--overwrite"]) == 0


def test_cli_parquet_input_and_output(tmp_path, monkeypatch):
    from conftest import write_rds

    monkeypatch.setenv("TRACE_RSCRIPT", str(tmp_path / "no_R_here"))
    rds = write_rds(DATA / "ai_treated.csv", tmp_path / "t.rds")
    pq = tmp_path / "t.parquet"
    assert main(["-q", "convert", str(rds), str(pq)]) == 0
    out = tmp_path / "o.parquet"
    assert main(["-q", "siteindex", "--treated", str(pq), "--control", str(DATA / "ai_control.csv"),
                 "--method", "m1A", "-o", str(out)]) == 0
    assert "sumNuc.x" in pl.read_parquet(out).columns


def test_cli_batch_continues_after_failure(tmp_path):
    sheet = tmp_path / "sheet.csv"
    pl.DataFrame({
        "treated": [str(DATA / "ai_treated.csv"), "missing.rds", str(DATA / "m5c_treated.csv")],
        "control": [str(DATA / "ai_control.csv"), "missing.rds", str(DATA / "m5c_control.csv")],
        "method": ["pseudo", "AI", "m5C"],
        "cov_threshold": [10, None, 20],
        "output": [None, None, "m5c/out.csv"],
    }).write_csv(sheet)
    rc = main(["-q", "batch", str(sheet), "--outdir", str(tmp_path / "out"),
               "--reference", str(DATA / "transcripts.fa"), "--always-keep-chr", "spikeIn"])
    assert rc == 1  # one row failed
    assert (tmp_path / "out" / "ai_treated_pseudo_siteindex.csv").exists()
    assert (tmp_path / "m5c" / "out.csv").exists()


def test_cli_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    pl.DataFrame({"chr": ["a"], "gencoor": [1], "refSeq": ["A"]}).write_csv(bad)
    assert main(["-q", "siteindex", "--treated", str(bad), "--control", str(bad),
                 "--method", "AI", "-o", str(tmp_path / "o.csv")]) == 1


@pytest.mark.parametrize("method", ["AI", "ac4C_CCG", "m5C_CC"])
def test_batching_gives_identical_results(method, tmp_path):
    from trace_rna import write_siteindex

    prefix = "m5c" if method.startswith("m5C") else "ai"
    res = Resources.load(snps=DATA / "snps.csv", reference=DATA / "transcripts.fa",
                         always_keep_chr=["spikeIn"])
    kw = dict(cov_threshold=20, resources=res, experiment="e")
    t, c = DATA / f"{prefix}_treated.csv", DATA / f"{prefix}_control.csv"
    whole = compute_siteindex(t, c, method, chunk_rows=None, **kw)
    batched = compute_siteindex(t, c, method, chunk_rows=100, **kw)  # ~1 transcript per batch
    assert batched.equals(whole)
    assert whole["pos"].to_list() == sorted(  # sorted by transcript, then position
        whole["pos"].to_list(), key=lambda p: (p.rsplit("_", 1)[0], int(p.rsplit("_", 1)[1])))

    for suffix in ("parquet", "csv.gz"):
        out = tmp_path / f"o.{suffix}"
        n = write_siteindex(t, c, method, out, chunk_rows=100, **kw)
        back = pl.read_parquet(out) if suffix == "parquet" else pl.read_csv(out, null_values="NA")
        assert n == whole.height == back.height
        assert back["pos"].to_list() == whole["pos"].to_list()
    assert not list(tmp_path.glob(".*part*"))  # temporary files cleaned up


def test_csv_is_r_style(tmp_path):
    from trace_rna.io import write_table

    df = pl.DataFrame({"a": [3.0, 0.5, float("inf"), float("nan"), None], "b": [True, False, None, True, True]})
    write_table(df, tmp_path / "x.csv")
    assert (tmp_path / "x.csv").read_text().splitlines() == ["a,b", "3,TRUE", "0.5,FALSE", "Inf,NA", "NA,TRUE", "NA,TRUE"]


def test_coverage_threshold_is_inclusive():
    """Sites with coverage exactly at the threshold are kept (>= 20)."""
    x = _sample({"refSeq": ["A", "A", "A"], "A": [19, 20, 21]})
    out = compute_siteindex(x, x, "AI", cov_threshold=20)
    assert out["cov.x"].to_list() == [20, 21]
