from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from trace_rna import Resources, compute_geneindex
from trace_rna.cli import main

DATA = Path(__file__).parent / "data"


def _sample(rows: list[tuple[str, int, str, int, int]]) -> pl.DataFrame:
    """rows: (chr, gencoor, refSeq, A count, G count); coverage = A + G."""
    return pl.DataFrame(
        {
            "chr": [r[0] for r in rows], "gencoor": [r[1] for r in rows], "refSeq": [r[2] for r in rows],
            "cov": [r[3] + r[4] for r in rows], "A": [r[3] for r in rows], "G": [r[4] for r in rows],
            "C": [0] * len(rows), "T": [0] * len(rows), "-": [0] * len(rows),
        }
    )


def test_sums_and_no_coverage_filter():
    x = _sample([
        ("geneA", 1, "A", 5, 5),      # low per-site coverage still counts for genes
        ("geneA", 2, "A", 140, 0),
        ("geneA", 3, "C", 999, 0),    # not an A site -> ignored
        ("geneB", 1, "A", 90, 10),    # geneB: low coverage (100) but kept
        ("geneC", 1, "A", 0, 0),      # geneC: no reads -> undefined score -> dropped
    ])
    y = _sample([("geneA", 1, "A", 150, 0), ("geneA", 2, "A", 150, 0),
                 ("geneB", 1, "A", 100, 0), ("geneC", 1, "A", 0, 0)])
    out = compute_geneindex(x, y, "AI", resources=Resources())
    # no coverage threshold: geneB (coverage 100) is kept; geneC (no reads, 0/0) is dropped
    assert out["chr.x"].to_list() == ["geneA", "geneB"]
    row = out.row(0, named=True)
    assert (row["sumA.x"], row["sumG.x"], row["sumCov.x"]) == (145, 5, 150)
    assert row["geneindex.x"] == pytest.approx(5 / 150)
    assert row["geneindex.y"] == 0 and row["log2FC"] == float("inf")
    assert out.columns[:6] == ["chr.x", "sumA.x", "sumC.x", "sumG.x", "sumT.x", "sumCov.x"]


def test_only_shared_sites_are_summed():
    x = _sample([("g", 1, "A", 100, 100), ("g", 2, "A", 100, 100)])
    y = _sample([("g", 1, "A", 200, 0)])  # position 2 missing in control
    out = compute_geneindex(x, y, "AI")
    assert out["sumCov.x"].item() == 200


def test_method_specific_columns():
    x = _sample([("g", 1, "A", 100, 50)])
    assert "sumNuc.x" in compute_geneindex(x, x, "m1A").columns
    t = _sample([("g", 1, "T", 0, 0)]).with_columns(pl.lit(80).alias("T"), pl.lit(20).alias("-"), pl.lit(100).alias("cov"))
    out = compute_geneindex(t, t, "pseudo")
    assert out["sumDeletion.x"].item() == 20 and out["geneindex.x"].item() == pytest.approx(0.2)


def test_batching_identical():
    res = Resources.load(snps=DATA / "snps.csv")
    kw = dict(resources=res)
    t, c = DATA / "ai_treated.csv", DATA / "ai_control.csv"
    assert compute_geneindex(t, c, "ac4C_CCG", chunk_rows=100, **kw).equals(
        compute_geneindex(t, c, "ac4C_CCG", chunk_rows=None, **kw))


def test_cli_geneindex(tmp_path):
    out = tmp_path / "genes.csv"
    rc = main(["-q", "geneindex", "--treated", str(DATA / "ai_treated.csv"),
               "--control", str(DATA / "ai_control.csv"), "--method", "AI",
               "--snps", str(DATA / "snps.csv"), "--experiment", "exp1", "-o", str(out)])
    assert rc == 0
    df = pl.read_csv(out, null_values="NA")
    assert df.height > 0 and df["experiment"].unique().to_list() == ["exp1"]


def test_cli_batch_mixed_levels(tmp_path):
    sheet = tmp_path / "sheet.tsv"
    pl.DataFrame({
        "treated": [str(DATA / "ai_treated.csv")] * 2,
        "control": [str(DATA / "ai_control.csv")] * 2,
        "method": ["AI", "AI"],
        "level": ["site", "gene"],
    }).write_csv(sheet, separator="\t")
    assert main(["-q", "batch", str(sheet), "--outdir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "ai_treated_AI_siteindex.csv").exists()
    genes = pl.read_csv(tmp_path / "out" / "ai_treated_AI_geneindex.csv", null_values="NA")
    assert genes.height > 0


def test_cli_batch_rejects_bad_level(tmp_path):
    sheet = tmp_path / "sheet.csv"
    pl.DataFrame({"treated": [str(DATA / "ai_treated.csv")], "control": [str(DATA / "ai_control.csv")],
                  "method": ["AI"], "level": ["region"]}).write_csv(sheet)
    assert main(["-q", "batch", str(sheet), "--outdir", str(tmp_path)]) == 1


EXAMPLE_HEADER = ("chr.x,sumA.x,sumC.x,sumG.x,sumT.x,sumCov.x,sumA.y,sumC.y,sumG.y,sumT.y,sumCov.y,"
                  "geneindex.x,geneindex.y,diff,log2FC,pVal,mlog10_pVal,experiment,datafile1,datafile2")


def test_geneindex_columns_match_example_output(tmp_path):
    """Columns as in the example R gene index output, without 'target', plus the two data file names."""
    out = tmp_path / "g.csv"
    assert main(["-q", "geneindex", "--treated", str(DATA / "ai_treated.csv"),
                 "--control", str(DATA / "ai_control.csv"), "--method", "AI", "-o", str(out)]) == 0
    assert out.read_text().splitlines()[0] == EXAMPLE_HEADER


def test_index_diff_log2fc_and_extreme_pvalues():
    """Gene index, diff and log2FC from summed counts; p-values down to the smallest doubles."""
    import numpy as np

    from trace_rna.stats import chisq_2x2_pvalue

    x = _sample([("g", 1, "A", 3000, 60), ("g", 2, "A", 1000, 0)])
    y = _sample([("g", 1, "A", 2000, 2), ("g", 2, "A", 500, 0)])
    row = compute_geneindex(x, y, "AI").row(0, named=True)
    gx, gy = 60 / 4060, 2 / 2502
    assert row["geneindex.x"] == pytest.approx(gx, rel=1e-12)
    assert row["diff"] == pytest.approx(gx - gy, rel=1e-12)
    assert row["log2FC"] == pytest.approx(np.log2(gx / gy), rel=1e-12)
    # very large differences: p-value underflows to exactly 0 (mlog10_pVal Inf) ...
    p, _ = chisq_2x2_pvalue(*(np.array([v]) for v in (4_000_000, 20_000, 500_000, 100)))
    assert p[0] == 0
    # ... while tiny but representable p-values are kept
    p, _ = chisq_2x2_pvalue(*(np.array([v]) for v in (1_000_000, 3_000, 500_000, 100)))
    assert 0 < p[0] < 1e-250


def test_old_command_names_are_gone(tmp_path):
    with pytest.raises(SystemExit):  # argparse: invalid choice
        main(["-q", "genescore", "--treated", str(DATA / "ai_treated.csv"),
              "--control", str(DATA / "ai_control.csv"), "--method", "AI", "-o", str(tmp_path / "g.csv")])


def test_geneindex_has_no_coverage_option(tmp_path):
    with pytest.raises(SystemExit):  # argparse rejects the option for geneindex
        main(["geneindex", "--treated", "a.rds", "--control", "b.rds", "--method", "AI",
              "--cov-threshold", "300", "-o", str(tmp_path / "x.csv")])
    with pytest.raises(TypeError):
        compute_geneindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI", cov_threshold=300)


@pytest.mark.parametrize("method", ["CU_AC", "CU_TA", "m3C", "pseudo_UNUA"])
def test_removed_methods_are_rejected(method):
    with pytest.raises(ValueError, match="Unknown method"):
        compute_geneindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", method)


def test_always_keep_chr_in_geneindex(tmp_path):
    x = _sample([("spikeIn", 1, "A", 90, 10), ("spikeIn", 2, "A", 80, 20),   # spike-in pos 2 only in treated
                 ("geneA", 1, "A", 50, 50), ("geneA", 2, "A", 40, 60),  # geneA pos 2 only in treated
                 ("onlyX", 1, "A", 70, 30)])                           # transcript only in treated
    y = _sample([("spikeIn", 1, "A", 100, 0), ("geneA", 1, "A", 100, 0)])

    plain = compute_geneindex(x, y, "AI")
    assert plain["chr.x"].to_list() == ["geneA", "spikeIn"]
    assert plain.filter(pl.col("chr.x") == "spikeIn")["sumCov.x"].item() == 100  # shared site only

    kept = compute_geneindex(x, y, "AI", resources=Resources(always_keep_chr=("spikeIn", "onlyX")))
    rows = {r["chr.x"]: r for r in kept.iter_rows(named=True)}
    assert set(rows) == {"spikeIn", "geneA", "onlyX"}
    assert rows["spikeIn"]["sumCov.x"] == 200 and rows["spikeIn"]["geneindex.x"] == pytest.approx(30 / 200)
    assert rows["geneA"]["sumCov.x"] == 100           # not kept -> unchanged
    assert rows["onlyX"]["geneindex.x"] == pytest.approx(0.3)
    assert rows["onlyX"]["geneindex.y"] is None and rows["onlyX"]["pVal"] is None


def test_cli_geneindex_accepts_always_keep_chr(tmp_path):
    out = tmp_path / "g.csv"
    assert main(["-q", "geneindex", "--treated", str(DATA / "m5c_treated.csv"), "--control",
                 str(DATA / "m5c_control.csv"), "--method", "AI", "--always-keep-chr", "spikeIn", "-o", str(out)]) == 0
    genes = pl.read_csv(out, null_values="NA")
    # the spike-in has positions missing in the control, so with the option its treated sum covers more sites
    without = compute_geneindex(DATA / "m5c_treated.csv", DATA / "m5c_control.csv", "AI")
    spike = lambda df: df.filter(pl.col("chr.x") == "spikeIn")["sumCov.x"].item()  # noqa: E731
    assert spike(genes) > spike(without)
