"""Gene index: treated (``.x``) versus control (``.y``) sample.

No coverage threshold is applied:

1. Keep sites whose reference base matches the method (original base and motif
   where the method has them) and that are not in the SNP list.
2. Keep sites present in both samples.
3. Per gene (``chr``) and sample, sum the nucleotide counts and coverage over
   those sites. Keep every gene whose index can be computed in both samples
   (genes with no modified or unmodified reads at all give 0/0 and are dropped).
4. Gene index = ``sum(modified) / (sum(modified) + sum(unmodified))`` (column
   ``geneindex.x``/``geneindex.y``), plus
   ``diff``, ``log2FC`` and a chi-square p-value on the summed counts
   (2x2 chi-square test).
"""

from __future__ import annotations

import logging
from typing import Iterator

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
from .io import TableWriter
from .methods import Method, get_method
from .stats import chisq_2x2_pvalue

log = logging.getLogger(__name__)

#: Summed columns, in output order (sumDeletion only for deletion methods).
_SUMS = [("A", "sumA"), ("C", "sumC"), ("G", "sumG"), ("T", "sumT"), ("cov", "sumCov")]


def _sum_columns(m: Method) -> list[tuple[str, str]]:
    cols = list(_SUMS)
    if "deletion" in m.count_columns:
        cols.append(("deletion", "sumDeletion"))
    return cols


def _index_genes(
    x: pl.DataFrame, y: pl.DataFrame, m: Method, res: Resources, labels: tuple[str | None, str | None, str | None]
) -> pl.DataFrame:
    shared = x.select(_KEY).join(y.select(_KEY), on=_KEY, how="inner")
    sums = _sum_columns(m)
    keep = _in_chr(res.always_keep_chr)

    def per_gene(df: pl.DataFrame, s: str) -> pl.DataFrame:
        # Sites present in both samples; for always-kept transcripts all sites.
        in_shared = df.join(shared, on=_KEY, how="semi")
        if res.always_keep_chr:
            in_shared = pl.concat([in_shared, df.filter(keep).join(shared, on=_KEY, how="anti")])
        return (
            in_shared
            .group_by("chr")
            .agg(
                *(pl.col(src).sum().alias(dst + s) for src, dst in sums),
                pl.sum_horizontal(m.modified).sum().alias(f"_mod{s}"),
                pl.sum_horizontal(m.unmodified).sum().alias(f"_unmod{s}"),
            )
        )

    # Genes in both samples; always-kept transcripts even if only one sample has them.
    genes = per_gene(x, ".x").join(per_gene(y, ".y"), on="chr", how="full", coalesce=True)
    in_both = pl.col("_mod.x").is_not_null() & pl.col("_mod.y").is_not_null()
    genes = genes.filter(in_both | keep)
    genes = genes.with_columns(
        *(
            (pl.col(f"_mod{s}") / (pl.col(f"_mod{s}") + pl.col(f"_unmod{s}"))).alias(f"geneindex{s}")
            for s in (".x", ".y")
        )
    ).filter(
        (pl.col("geneindex.x").is_not_nan() & pl.col("geneindex.y").is_not_nan()).fill_null(False)
        | keep
    ).sort("chr")

    pval, is_na = chisq_2x2_pvalue(
        *(genes[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()
          for c in ("_unmod.x", "_mod.x", "_unmod.y", "_mod.y"))
    )
    genes = genes.with_columns(
        pl.Series("pVal", pval).set(pl.Series(is_na), None),
        *([pl.col("_mod.x").alias("sumNuc.x"), pl.col("_mod.y").alias("sumNuc.y")] if m.report_sum else []),
    ).with_columns(
        (-pl.col("pVal").log10()).alias("mlog10_pVal"),
        (pl.col("geneindex.x") / pl.col("geneindex.y")).log(2).alias("log2FC"),
        (pl.col("geneindex.x") - pl.col("geneindex.y")).alias("diff"),
        *label_columns(*labels),
    )
    return genes.select(
        pl.col("chr").alias("chr.x"),
        *(dst + ".x" for _, dst in sums),
        *(dst + ".y" for _, dst in sums),
        *(["sumNuc.x", "sumNuc.y"] if m.report_sum else []),
        "geneindex.x", "geneindex.y", "diff", "log2FC", "pVal", "mlog10_pVal",
        *LABEL_COLUMNS,
    )


def iter_geneindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    *,
    resources: Resources | None = None,
    experiment: str | None = None,
    chunk_rows: int | None = DEFAULT_CHUNK_ROWS,
) -> Iterator[pl.DataFrame]:
    """Like :func:`compute_geneindex`, yielding batches of genes in sorted order."""
    m = get_method(method) if isinstance(method, str) else method
    res = resources or Resources()
    if chunk_rows is not None and chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")

    x_lf, y_lf = _open_sample(treated, m), _open_sample(control, m)
    check_resources(x_lf, res, m)
    batches = ([None] if chunk_rows is None else _chrom_batches([x_lf, y_lf], chunk_rows)) or [None]
    n_genes = 0
    for i, chroms in enumerate(batches, start=1):
        x = _prepare_sample(x_lf, m, None, res, "treated", chroms)
        y = _prepare_sample(y_lf, m, None, res, "control", chroms)
        batch = _index_genes(x, y, m, res, (experiment, data_file_name(treated), data_file_name(control)))
        del x, y
        n_genes += batch.height
        log.info("batch %d/%d: %d genes (total %d)", i, len(batches), batch.height, n_genes)
        yield batch


def compute_geneindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    *,
    resources: Resources | None = None,
    experiment: str | None = None,
    chunk_rows: int | None = DEFAULT_CHUNK_ROWS,
) -> pl.DataFrame:
    """Compare a treated sample (``.x``) with a control sample (``.y``) gene by gene.

    Args:
        treated, control: Count tables (paths or polars frames).
        method: Method name (see ``trace methods``) or :class:`Method`.
        resources: SNP list and reference bases.
        experiment: Label written to the ``experiment`` column. The columns
            ``datafile1``/``datafile2`` get the file names of ``treated``/``control``.
        chunk_rows: Work on batches of whole genes of about this many input rows.

    Returns:
        One row per gene (``chr.x``), sorted by name, with the summed counts,
        ``geneindex.x/.y``, ``diff``, ``log2FC``, ``pVal``, ``mlog10_pVal``,
        ``experiment``, ``datafile1`` and ``datafile2``.
    """
    parts = list(iter_geneindex(
        treated, control, method, resources=resources,
        experiment=experiment, chunk_rows=chunk_rows,
    ))
    return pl.concat(parts, how="vertical_relaxed")


def write_geneindex(
    treated: TableLike,
    control: TableLike,
    method: str | Method,
    output: PathLike,
    *,
    overwrite: bool = False,
    **kwargs,
) -> int:
    """Compute the gene index and write it batch by batch; returns the number of genes."""
    res = kwargs.get("resources")
    check_output_not_input(output, treated, control,
                           *(getattr(res, "source_files", ()) if res is not None else ()))
    with TableWriter(output, overwrite=overwrite) as writer:
        for batch in iter_geneindex(treated, control, method, **kwargs):
            writer.write(batch)
        return writer.rows

