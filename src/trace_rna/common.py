"""Shared steps of site and gene index: opening count tables, per-site filtering, batching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import polars as pl

from .io import COUNT_COLUMNS, normalise_bases, read_counts, read_positions
from .methods import Method
from .reference import ReferenceFasta, is_fasta

log = logging.getLogger(__name__)

PathLike = Union[str, Path]
TableLike = Union[PathLike, pl.DataFrame, pl.LazyFrame]

_KEY = ["chr", "gencoor"]


@dataclass(frozen=True)
class Resources:
    """Annotation shared by many sample pairs; load it once with :meth:`load`.

    Attributes:
        snps: Positions to exclude (e.g. known SNPs).
        reference: Original (unconverted) reference, needed by methods on
            converted references (m5C, m6A): the FASTA of the original
            transcriptome.
        always_keep_chr: Chromosomes/transcripts (e.g. a spike-in such as
            ``MS2``) that bypass the coverage filter and are kept even if only
            one sample covers them.
    """

    snps: pl.DataFrame | None = None
    reference: ReferenceFasta | None = None
    always_keep_chr: tuple[str, ...] = field(default_factory=tuple)
    source_files: tuple[Path, ...] = field(default_factory=tuple, compare=False)

    @classmethod
    def load(
        cls,
        snps: TableLike | None = None,
        reference: PathLike | ReferenceFasta | None = None,
        always_keep_chr: Sequence[str] = (),
    ) -> "Resources":
        if reference is None or isinstance(reference, ReferenceFasta):
            ref = reference
        elif is_fasta(reference):
            ref = ReferenceFasta.read(reference)
        else:
            raise ValueError(f"--reference must be a FASTA file (.fa, .fasta, .fna, optionally .gz): {reference}")
        files = tuple(Path(f) for f in (snps, reference) if isinstance(f, (str, Path)))
        return cls(
            source_files=files,
            snps=_load_positions(snps),
            reference=ref,
            always_keep_chr=tuple(always_keep_chr),
        )


def _load_positions(source: TableLike | None, values: Sequence[str] = ()) -> pl.DataFrame | None:
    if source is None:
        return None
    if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
        df = source.lazy().collect()
        missing = [c for c in [*_KEY, *values] if c not in df.columns]
        if missing:
            raise ValueError(f"Position table lacks column(s) {missing}")
        return df.select(*_KEY, *values)
    return read_positions(source, values).collect()


#: Default number of input rows (both samples' largest) processed per batch.
#: ~5 million rows keeps peak memory around 1-2 GB for typical tables.
DEFAULT_CHUNK_ROWS = 5_000_000


def check_output_not_input(output: PathLike, *inputs: object) -> None:
    """Refuse to write a result onto one of the input files (that would destroy the input)."""
    out = Path(output).resolve()
    for src in inputs:
        if isinstance(src, (str, Path)) and Path(src).resolve() == out:
            raise ValueError(f"The output file {Path(output).name} is also an input file; "
                             "choose another name with -o so the input is not overwritten.")


#: Label columns at the end of every result, in this order.
LABEL_COLUMNS = ("experiment", "datafile1", "datafile2")


def data_file_name(source: TableLike) -> str | None:
    """File name (without folder) of a count table given as a path; None for in-memory tables."""
    return Path(source).name if isinstance(source, (str, Path)) else None


def label_columns(experiment: str | None, datafile1: str | None, datafile2: str | None) -> list[pl.Expr]:
    """Constant columns: experiment label and the names of the treated / control files."""
    return [
        pl.lit(experiment, dtype=pl.String).alias("experiment"),
        pl.lit(datafile1, dtype=pl.String).alias("datafile1"),
        pl.lit(datafile2, dtype=pl.String).alias("datafile2"),
    ]


def _in_chr(chroms: Sequence[str]) -> pl.Expr:
    return pl.col("chr").is_in(list(chroms)) if chroms else pl.lit(False)


def _open_sample(source: TableLike, method: Method) -> pl.LazyFrame:
    """Open a count table lazily (an .rds is converted to Parquet first)."""
    columns = list(COUNT_COLUMNS) + [c for c in method.count_columns if c not in COUNT_COLUMNS]
    if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
        lf = source.lazy()
        if "-" in lf.collect_schema().names():
            lf = lf.rename({"-": "deletion"})
        return lf.select(
            pl.col("chr").cast(pl.String), pl.col("gencoor").cast(pl.Int64),
            pl.col("cov").cast(pl.Float64), normalise_bases(pl.col("refSeq")).alias("refSeq"),
            *(pl.col(c).cast(pl.Float64) for c in columns),
        )
    return read_counts(source, columns)


def _prepare_sample(
    lf: pl.LazyFrame,
    method: Method,
    cov_threshold: float | None,
    res: Resources,
    label: str,
    chroms: Sequence[str] | None,
) -> pl.DataFrame:
    """Filter one sample down to the candidate sites.

    Keeps sites with the method's reference base (and original base / motif),
    coverage of at least ``cov_threshold`` (``None``: no per-site coverage filter)
    unless on an always-kept chromosome, and not in the SNP list.
    """
    columns = [c for c in lf.collect_schema().names() if c not in ("chr", "gencoor", "cov", "refSeq")]
    if chroms is not None:
        lf = lf.filter(pl.col("chr").is_in(list(chroms)))
    reference: pl.LazyFrame | None = None
    if method.needs_reference:
        if res.reference is None:
            raise ValueError(
                f"Method {method.name!r} needs the original (unconverted) reference: "
                "--reference transcriptome.fa"
            )
        if chroms is None:
            chroms = lf.select(pl.col("chr").unique()).collect(engine="streaming")["chr"].to_list()
        reference = res.reference.lazy_table(chroms)  # only the transcripts of this batch
        lf = lf.join(reference, on=_KEY, how="inner")

    sites = lf.filter(pl.col("refSeq") == method.ref_base)
    if method.reference_base is not None:
        sites = sites.filter(pl.col("base") == method.reference_base)

    # Motif: semi-join on neighbours with an allowed base. Neighbours are found
    # by coordinate (gencoor +/- offset on the same chr), not by row order, so
    # transcript ends and missing rows cannot pull in an unrelated base. Motifs
    # on the original base use the reference table, which is complete.
    for offset, allowed in sorted(method.motif.items()):
        col = method.motif_column
        lookup = reference if col == "base" else lf
        neighbours = lookup.filter(pl.col(col).is_in(sorted(allowed))).select(
            "chr", (pl.col("gencoor") - offset).alias("gencoor")
        )
        sites = sites.join(neighbours, on=_KEY, how="semi")

    if cov_threshold is not None:
        sites = sites.filter((pl.col("cov") >= cov_threshold) | _in_chr(res.always_keep_chr))
    if res.snps is not None:
        sites = sites.join(res.snps.lazy().select(_KEY), on=_KEY, how="anti")

    df = sites.select("chr", "gencoor", "cov", "refSeq", *columns).collect(engine="streaming")

    dup = df.select(pl.struct(_KEY).is_duplicated().sum()).item()
    if dup:
        raise ValueError(f"{label}: {dup} rows share a chr/gencoor position; positions must be unique.")
    log.debug("%s: %d candidate sites in this batch", label, df.height)
    return df


def _chrom_batches(samples: Sequence[pl.LazyFrame], chunk_rows: int) -> list[list[str]]:
    """Group chromosomes (sorted by name) into batches of at most ~chunk_rows rows."""
    sizes = (
        pl.concat([lf.group_by("chr").len() for lf in samples])
        .group_by("chr").agg(pl.col("len").max())
        .sort("chr")
        .collect(engine="streaming")
    )
    batches: list[list[str]] = []
    current: list[str] = []
    rows = 0
    for chr_, n in sizes.iter_rows():
        if current and rows + n > chunk_rows:
            batches.append(current)
            current, rows = [], 0
        current.append(chr_)
        rows += n
    if current:
        batches.append(current)
    return batches


#: Warn when more than this fraction of annotation positions (on transcripts that
#: are in the data) lie beyond the end of their transcript.
OUT_OF_RANGE_WARN_FRACTION = 0.5


def check_annotation_matches(
    data: pl.LazyFrame,
    annotation: pl.DataFrame | None,
    label: str,
    consequence: str,
) -> dict[str, int] | None:
    """Warn if an annotation table (SNP list, reference bases) does not fit the data.

    Two checks, logged as warnings:

    * none of the annotation's transcript names occur in the data (different
      transcriptome / annotation, or genomic instead of transcript names);
    * most annotation positions on shared transcripts lie beyond the last
      position of that transcript in the data (different coordinate system,
      e.g. genome coordinates instead of transcript coordinates).

    ``label`` names the file in the message (e.g. "SNP file (--snps)"),
    ``consequence`` says what a mismatch means for the result.
    Returns the numbers used, or None if there is no annotation.
    """
    if annotation is None or annotation.height == 0:
        return None
    lengths = data.group_by("chr").agg(pl.col("gencoor").max().alias("_end")).collect(engine="streaming")
    joined = annotation.select("chr", "gencoor").join(lengths, on="chr", how="left")
    n_total = joined.height
    n_on_data = joined.filter(pl.col("_end").is_not_null()).height
    n_beyond = joined.filter(pl.col("gencoor") > pl.col("_end")).height
    stats = {"positions": n_total, "on_data_transcripts": n_on_data, "beyond_transcript_end": n_beyond}

    if n_on_data == 0:
        example_ann = sorted(annotation["chr"].unique().head(3).to_list())
        example_data = sorted(lengths["chr"].head(3).to_list())
        log.warning(
            "%s does not fit the data: none of its transcript names occur in the count table "
            "(file: %s; data: %s). The file and the data probably use a different "
            "reference/annotation. %s",
            label, ", ".join(example_ann), ", ".join(example_data), consequence,
        )
    elif n_beyond > OUT_OF_RANGE_WARN_FRACTION * n_on_data:
        log.warning(
            "%s does not fit the data: %d of %d positions on matching transcripts lie beyond "
            "the transcript end. The file probably uses different coordinates (e.g. genome "
            "instead of transcript positions, or another genome build). %s",
            label, n_beyond, n_on_data, consequence,
        )
    else:
        log.info("%s: %d of %d positions lie on transcripts in the data", label, n_on_data, n_total)
    return stats


def check_fasta_matches(data: pl.LazyFrame, fasta: ReferenceFasta, label: str, consequence: str) -> dict[str, int]:
    """Warn if the reference FASTA does not fit the data (names or transcript lengths)."""
    lengths = data.group_by("chr").agg(pl.col("gencoor").max().alias("_end")).collect(engine="streaming")
    joined = lengths.join(fasta.lengths(), on="chr", how="left")
    n_data = joined.height
    n_found = joined.filter(pl.col("length").is_not_null()).height
    n_longer = joined.filter(pl.col("_end") > pl.col("length")).height
    stats = {"data_transcripts": n_data, "in_fasta": n_found, "data_longer_than_fasta": n_longer}
    if n_found == 0:
        log.warning(
            "%s does not fit the data: none of the count table's transcripts are in the FASTA "
            "(data: %s; FASTA: %s). Check that both use the same transcriptome. %s",
            label, ", ".join(sorted(lengths["chr"].head(3).to_list())),
            ", ".join(fasta.names[:3]), consequence,
        )
    elif n_longer > OUT_OF_RANGE_WARN_FRACTION * n_found:
        log.warning(
            "%s does not fit the data: for %d of %d transcripts the count table has positions "
            "beyond the end of the FASTA sequence. The FASTA is probably a different "
            "version/build of the transcriptome. %s", label, n_longer, n_found, consequence,
        )
    else:
        if n_found < n_data:
            log.info("%s: %d of %d transcripts of the data are in the FASTA; the others are skipped",
                     label, n_found, n_data)
        else:
            log.info("%s: all %d transcripts of the data are in the FASTA", label, n_data)
    return stats


def check_resources(data: pl.LazyFrame, res: "Resources", method: Method) -> None:
    """Check that the SNP list and, if used, the reference fit the data (warnings only)."""
    check_annotation_matches(data, res.snps, "SNP file (--snps)",
                             "SNPs will not be removed correctly.")
    if method.needs_reference:
        consequence = f"Method {method.name} will find few or no sites."
        if res.reference is not None:
            check_fasta_matches(data, res.reference, "Reference FASTA (--reference)", consequence)
