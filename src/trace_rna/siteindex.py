"""Site index: treated (``.x``) versus control (``.y``) sample.

Pipeline for each pair of samples:

1. Keep sites whose reference base matches the method (and, for converted
   references such as bisulfite, whose original base matches), optionally
   within a sequence motif.
2. Keep sites with coverage of at least the threshold (default 20) and not in the SNP list.
3. Keep sites present in both samples.
4. Per site and sample: site index = ``modified / (modified + unmodified)``
   (columns ``siteindex.x``/``siteindex.y``).
   Between samples: ``diff``, ``log2FC``, and a chi-square p-value
   (2x2 chi-square test).
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np
import polars as pl

from .common import (
    _KEY,
    DEFAULT_CHUNK_ROWS,
    PathLike,
    Resources,
    TableLike,
    _chrom_batches,
    _in_chr,
    LABEL_COLUMNS,
    _open_sample,
    check_output_not_input,
    data_file_name,
    label_columns,
    check_resources,
    _prepare_sample,
)
from .io import COUNT_COLUMNS, TableWriter
from .methods import Method, get_method
from .stats import chisq_2x2_pvalue

log = logging.getLogger(__name__)

def _index_pair(x: pl.DataFrame, y: pl.DataFrame, m: Method, res: Resources, labels: tuple[str | None, str | None, str | None]) -> pl.DataFrame:
    """Steps 3-4: match sites between samples and compute the site index and statistics."""
    shared = x.select(_KEY).join(y.select(_KEY), on=_KEY, how="inner")

    def restrict(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
        kept = df.join(shared, on=_KEY, how="semi")
        if res.always_keep_chr:
            extra = df.filter(_in_chr(res.always_keep_chr)).join(shared, on=_KEY, how="anti")
            kept = pl.concat([kept, extra])
        return kept.select(
            pl.col("chr").alias("_chr"),
            pl.col("gencoor").alias("_gc"),
            pl.all().name.suffix(suffix),
        )

    merged = restrict(x, ".x").join(restrict(y, ".y"), on=["_chr", "_gc"], how="full", coalesce=True)
    merged = merged.sort(["_chr", "_gc"])

    def total(cols: Sequence[str], s: str) -> pl.Expr:
        return pl.sum_horizontal([pl.col(c + s) for c in cols], ignore_nulls=False)

    merged = merged.with_columns(
        *(total(m.modified, s).alias(f"_mod{s}") for s in (".x", ".y")),
        *(total(m.unmodified, s).alias(f"_unmod{s}") for s in (".x", ".y")),
    ).with_columns(
        *(
            (pl.col(f"_mod{s}") / (pl.col(f"_mod{s}") + pl.col(f"_unmod{s}"))).alias(f"siteindex{s}")
            for s in (".x", ".y")
        )
    )
    if m.control_missing_as_zero:
        y_score = pl.col("siteindex.y")
        merged = merged.with_columns(
            pl.when(y_score.is_null() | y_score.is_nan()).then(0.0).otherwise(y_score).alias("siteindex.y")
        )

    pval, is_na = chisq_2x2_pvalue(
        *(merged[c].to_numpy().astype(np.float64) for c in ("_unmod.x", "_mod.x", "_unmod.y", "_mod.y"))
    )
    merged = merged.with_columns(
        pl.Series("pVal", pval).set(pl.Series(is_na), None),
        *([pl.col("_mod.x").alias("sumNuc.x"), pl.col("_mod.y").alias("sumNuc.y")] if m.report_sum else []),
    ).with_columns(
        (-pl.col("pVal").log10()).alias("mlog10_pVal"),
        (pl.col("siteindex.x") / pl.col("siteindex.y")).log(2).alias("log2FC"),
        (pl.col("siteindex.x") - pl.col("siteindex.y")).alias("diff"),
        *label_columns(*labels),
        pl.concat_str(pl.col("_chr"), pl.col("_gc").cast(pl.String), separator="_").alias("pos"),
    )

    sample_cols = ["chr", "gencoor", "cov", "refSeq", *COUNT_COLUMNS]
    if "deletion" in m.count_columns:
        sample_cols.append("deletion")
    return merged.select(
        "pos",
        *(c + ".x" for c in sample_cols),
        *(c + ".y" for c in sample_cols),
        *(["sumNuc.x", "sumNuc.y"] if m.report_sum else []),
        "siteindex.x", "siteindex.y", "pVal", "mlog10_pVal", "log2FC", "diff",
        *LABEL_COLUMNS,
    )


def iter_siteindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    *,
    cov_threshold: float = 20,
    resources: Resources | None = None,
    experiment: str | None = None,
    chunk_rows: int | None = DEFAULT_CHUNK_ROWS,
) -> Iterator[pl.DataFrame]:
    """Like :func:`compute_siteindex`, but yields the result in batches of
    whole chromosomes/transcripts, in sorted order, to bound memory use.

    ``chunk_rows=None`` processes everything in one batch.
    """
    m = get_method(method) if isinstance(method, str) else method
    res = resources or Resources()
    if cov_threshold < 0:
        raise ValueError("cov_threshold must be >= 0")
    if chunk_rows is not None and chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")

    x_lf, y_lf = _open_sample(treated, m), _open_sample(control, m)
    check_resources(x_lf, res, m)
    batches: list[list[str] | None] = (
        [None] if chunk_rows is None else _chrom_batches([x_lf, y_lf], chunk_rows)
    ) or [None]
    n_sites = 0
    for i, chroms in enumerate(batches, start=1):
        x = _prepare_sample(x_lf, m, cov_threshold, res, "treated", chroms)
        y = _prepare_sample(y_lf, m, cov_threshold, res, "control", chroms)
        batch = _index_pair(x, y, m, res, (experiment, data_file_name(treated), data_file_name(control)))
        del x, y
        n_sites += batch.height
        log.info("batch %d/%d: %d sites (total %d)", i, len(batches), batch.height, n_sites)
        yield batch


def compute_siteindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    *,
    cov_threshold: float = 20,
    resources: Resources | None = None,
    experiment: str | None = None,
    chunk_rows: int | None = DEFAULT_CHUNK_ROWS,
) -> pl.DataFrame:
    """Compare a treated sample (``.x``) with a control sample (``.y``) site by site.

    Args:
        treated: Count table (path or polars frame) of the sample expected to
            carry the modification, e.g. +enzyme or +MS2 loop.
        control: Count table of the comparison sample.
        method: Method name (see ``trace methods``) or :class:`Method`.
        cov_threshold: Keep sites with coverage of at least this (default 20).
        resources: SNP list, reference bases, spike-ins.
        experiment: Label written to the ``experiment`` column. The columns
            ``datafile1``/``datafile2`` get the file names of ``treated``/``control``.
        chunk_rows: Work on batches of whole transcripts of about this many
            input rows. The result is the same; only peak memory changes.

    Returns:
        One row per site, sorted by chr and position, with the columns
        described in the README. For very large results use :func:`write_siteindex`,
        which never holds the whole result in memory.
    """
    parts = list(iter_siteindex(
        treated, control, method, cov_threshold=cov_threshold, resources=resources,
        experiment=experiment, chunk_rows=chunk_rows,
    ))
    return pl.concat(parts, how="vertical_relaxed")


def write_siteindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    output: PathLike,
    *,
    overwrite: bool = False,
    **kwargs,
) -> int:
    """Compute the site index and write it batch by batch; returns the number of sites.

    ``output`` may be .csv, .csv.gz, .tsv or .parquet. Keyword arguments are
    passed to :func:`iter_siteindex`.
    """
    res = kwargs.get("resources")
    check_output_not_input(output, treated, control,
                           *(getattr(res, "source_files", ()) if res is not None else ()))
    with TableWriter(output, overwrite=overwrite) as writer:
        for batch in iter_siteindex(treated, control, method, **kwargs):
            writer.write(batch)
        return writer.rows
