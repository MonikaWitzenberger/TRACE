"""Warnings when the SNP / reference file does not fit the count tables."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from trace_rna import Resources, compute_geneindex, compute_siteindex
from trace_rna.common import check_annotation_matches

DATA = Path(__file__).parent / "data"
DATA_TX = pl.DataFrame({"chr": ["tx_1"] * 100 + ["tx_2"] * 50,
                        "gencoor": list(range(1, 101)) + list(range(1, 51))}).lazy()


def _pos(rows):
    return pl.DataFrame({"chr": [r[0] for r in rows], "gencoor": [r[1] for r in rows]})


def test_matching_file_gives_no_warning(caplog):
    with caplog.at_level(logging.INFO):
        stats = check_annotation_matches(DATA_TX, _pos([("tx_1", 5), ("tx_2", 50), ("other", 3)]), "SNP file", "x")
    assert stats == {"positions": 3, "on_data_transcripts": 2, "beyond_transcript_end": 0}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_different_transcript_names_warn(caplog):
    check_annotation_matches(DATA_TX, _pos([("chr1", 14653), ("chr2", 41241)]), "SNP file", "x")
    assert any("none of its transcript names" in r.message for r in caplog.records)


def test_genomic_coordinates_warn(caplog):
    check_annotation_matches(DATA_TX, _pos([("tx_1", 1_200_345), ("tx_2", 88_721), ("tx_1", 10)]), "SNP file", "x")
    assert any("beyond the transcript end" in r.message for r in caplog.records)


def test_few_out_of_range_positions_do_not_warn(caplog):
    check_annotation_matches(DATA_TX, _pos([("tx_1", 5), ("tx_1", 6), ("tx_1", 500)]), "SNP file", "x")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_warning_during_real_run(tmp_path, caplog):
    snps = tmp_path / "snps.csv"
    snps.write_text("pos\nchr1_14653\nchr1_16949\n")
    res = Resources.load(snps=snps)
    compute_geneindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI", resources=res)
    assert any("SNP file (--snps) does not fit" in r.message for r in caplog.records)


def test_reference_checked_only_for_methods_that_use_it(tmp_path, caplog):
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr1\nACGTACGT\n")
    res = Resources.load(reference=ref)
    compute_siteindex(DATA / "ai_treated.csv", DATA / "ai_control.csv", "AI", resources=res)
    assert not any("Reference FASTA" in r.message for r in caplog.records)
    compute_siteindex(DATA / "m5c_treated.csv", DATA / "m5c_control.csv", "m5C", resources=res)
    assert any("Reference FASTA (--reference) does not fit" in r.message for r in caplog.records)
