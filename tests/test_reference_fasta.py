"""Reading the reference FASTA of the original transcriptome (m5C, m5C_CC, m6A)."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from trace_rna import Resources, compute_geneindex, compute_siteindex
from trace_rna.cli import main
from trace_rna.reference import ReferenceFasta, is_fasta

DATA = Path(__file__).parent / "data"
CONVERTED = ["m5C", "m5C_CC"]


def _write_fasta(path: Path, *, crlf: bool = False, wrap: int = 60, lower: bool = False) -> Path:
    """The test FASTA rewritten in another layout (line endings, line width, case, gzip)."""
    seqs = ReferenceFasta.read(DATA / "transcripts.fa").sequences
    nl = "\r\n" if crlf else "\n"
    out = []
    for name, seq in seqs.items():
        seq = seq.decode()
        if lower:
            seq = seq[: len(seq) // 2].lower() + seq[len(seq) // 2:]
        out.append(f">{name} some description")
        out += [seq[i:i + wrap] for i in range(0, len(seq), wrap)]
    text = nl.join(out) + nl
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", newline="") as fh:
            fh.write(text)
    else:
        path.write_text(text, newline="")
    return path


@pytest.fixture(scope="module")
def fasta(tmp_path_factory):
    return _write_fasta(tmp_path_factory.mktemp("ref") / "transcriptome.fa.gz", crlf=True, wrap=70, lower=True)


@pytest.mark.parametrize("method", CONVERTED)
def test_fasta_layout_does_not_matter_site(method, fasta):
    t, c = DATA / "m5c_treated.csv", DATA / "m5c_control.csv"
    via_plain = compute_siteindex(t, c, method, resources=Resources.load(reference=DATA / "transcripts.fa"))
    via_fa = compute_siteindex(t, c, method, resources=Resources.load(reference=fasta))
    assert via_fa.equals(via_plain)


@pytest.mark.parametrize("method", CONVERTED)
def test_fasta_layout_does_not_matter_gene(method, fasta):
    t, c = DATA / "m5c_treated.csv", DATA / "m5c_control.csv"
    via_plain = compute_geneindex(t, c, method, resources=Resources.load(reference=DATA / "transcripts.fa"))
    via_fa = compute_geneindex(t, c, method, resources=Resources.load(reference=fasta))
    assert via_fa.equals(via_plain)


def test_fasta_batched_equals_single_pass(fasta):
    res = Resources.load(reference=fasta)
    t, c = DATA / "m5c_treated.csv", DATA / "m5c_control.csv"
    assert compute_siteindex(t, c, "m5C_CC", resources=res, chunk_rows=100).equals(
        compute_siteindex(t, c, "m5C_CC", resources=res, chunk_rows=None))


def test_fasta_parsing(tmp_path):
    fa = tmp_path / "x.fasta.gz"
    with gzip.open(fa, "wt", newline="") as fh:
        fh.write(">tx1 description here\r\nacgU\r\nNNac\r\n>tx_2\nGG\n")
    ref = ReferenceFasta.read(fa)
    assert ref.sequences == {"tx1": b"ACGTNNAC", "tx_2": b"GG"}
    assert ref.table(["tx_2", "missing"]).rows() == [("tx_2", 1, "G"), ("tx_2", 2, "G")]
    assert ref.table(["tx1"])["gencoor"].to_list() == list(range(1, 9))


def test_fasta_duplicate_name_rejected(tmp_path):
    fa = tmp_path / "dup.fa"
    fa.write_text(">a\nAC\n>a\nGG\n")
    with pytest.raises(ValueError, match="more than once"):
        ReferenceFasta.read(fa)


def test_is_fasta():
    assert all(is_fasta(n) for n in ["t.fa", "t.FASTA", "t.fna", "t.fa.gz"])
    assert not any(is_fasta(n) for n in ["t.csv", "t.csv.gz", "t.parquet"])


def test_cli_with_fasta(tmp_path, fasta):
    out = tmp_path / "m5c.csv"
    assert main(["-q", "siteindex", "--treated", str(DATA / "m5c_treated.csv"),
                 "--control", str(DATA / "m5c_control.csv"), "--method", "m5C",
                 "--reference", str(fasta), "--always-keep-chr", "spikeIn", "-o", str(out)]) == 0
    assert pl.read_csv(out, null_values="NA").height > 0


def test_fasta_mismatch_warns(tmp_path, caplog):
    fa = tmp_path / "other.fa"
    fa.write_text(">chr1\nACGT\n")
    compute_siteindex(DATA / "m5c_treated.csv", DATA / "m5c_control.csv", "m5C",
                      resources=Resources.load(reference=fa))
    assert any("Reference FASTA (--reference) does not fit" in r.message for r in caplog.records)


def test_short_fasta_warns(tmp_path, caplog):
    """Same names, but sequences much shorter than the data (another version/build)."""
    names = ReferenceFasta.read(DATA / "transcripts.fa").names
    fa = tmp_path / "short.fa"
    fa.write_text("".join(f">{c}\nACGTACGTAC\n" for c in names))
    compute_siteindex(DATA / "m5c_treated.csv", DATA / "m5c_control.csv", "m5C",
                      resources=Resources.load(reference=fa))
    assert any("beyond the end of the FASTA sequence" in r.message for r in caplog.records)


def test_reference_must_be_fasta(tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text("pos,base\ntx_alpha_1,C\n")
    with pytest.raises(ValueError, match="must be a FASTA"):
        Resources.load(reference=ref)


def test_m5c_finds_methylated_sites():
    """In the test data the treated sample has more unconverted C than the control."""
    out = compute_siteindex(DATA / "m5c_treated.csv", DATA / "m5c_control.csv", "m5C",
                            resources=Resources.load(reference=DATA / "transcripts.fa"))
    assert out.height > 0 and out["diff"].mean() > 0.2
