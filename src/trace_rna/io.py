"""Reading inputs and writing results.

Count tables can be Parquet (recommended: fast, and only the needed columns
and rows are read), CSV/TSV, or RDS. RDS files are converted to Parquet
automatically the first time they are used (see :mod:`trace_rna.rds`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

log = logging.getLogger(__name__)

#: Columns of the count table, and the name each one gets internally.
#: The per-position tables name the deletion column "-".
DELETION_SOURCE_COLUMN = "-"
BASE_COLUMNS = ("chr", "gencoor", "cov", "refSeq")
COUNT_COLUMNS = ("C", "T", "A", "G")

_PARQUET = {".parquet", ".pq"}
_TEXT = {".csv": ",", ".tsv": "\t", ".txt": "\t"}
#: Count tables in text format above this size get a one-time Parquet copy.
TEXT_TO_PARQUET_BYTES = 50_000_000


class InputError(ValueError):
    """An input file is missing, unreadable, or lacks required columns."""


def _suffix(path: Path) -> str:
    """File type suffix, ignoring a trailing compression suffix."""
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] in {".gz", ".bz2", ".zst"}:
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _check_exists(path: Path) -> None:
    if not path.is_file():
        raise InputError(f"File not found: {path}")


def _require(schema_names: Iterable[str], required: Iterable[str], path: Path) -> None:
    present = set(schema_names)
    missing = [c for c in required if c not in present]
    if missing and len(present) == 1 and ";" in next(iter(present)):
        raise InputError(
            f"{path.name} uses ';' as separator (as German Excel saves CSV files). "
            "Please save it with commas instead, e.g. in Excel as 'CSV UTF-8 (comma delimited)', "
            "or in R with write.csv() / data.table::fwrite()."
        )
    if missing:
        raise InputError(
            f"{path.name} is missing column(s) {missing}. "
            f"Columns found: {sorted(present)}"
        )


#: Column types fixed when reading text tables, so the type never depends on
#: the first rows of a large file (e.g. counts that are whole numbers early on).
_TEXT_TYPES: dict[str, type[pl.DataType]] = {
    "chr": pl.String, "pos": pl.String, "refSeq": pl.String, "base": pl.String, "gene": pl.String,
    "gencoor": pl.Float64, "cov": pl.Float64,  # gencoor may be written as 12 or 12.0
    "A": pl.Float64, "C": pl.Float64, "G": pl.Float64, "T": pl.Float64, "N": pl.Float64,
    "-": pl.Float64, ".": pl.Float64,
}


def _scan_text(path: Path, separator: str) -> pl.LazyFrame:
    """Lazily read CSV/TSV (optionally .gz), with fixed types for known columns."""
    header = pl.read_csv(path, separator=separator, n_rows=0).columns
    overrides = {c: t for c, t in _TEXT_TYPES.items() if c in header}
    if path.suffix.lower() == ".gz":  # compressed text cannot be scanned lazily
        return pl.read_csv(path, separator=separator, schema_overrides=overrides,
                           null_values=["NA"], infer_schema_length=10_000).lazy()
    return pl.scan_csv(path, separator=separator, schema_overrides=overrides,
                       null_values=["NA"], infer_schema_length=10_000)


def scan_table(path: str | Path) -> pl.LazyFrame:
    """Lazily open a Parquet, CSV/TSV or RDS table.

    An RDS file is converted to Parquet once (see :mod:`trace_rna.rds`) and
    the Parquet copy is read from then on.
    """
    path = Path(path)
    _check_exists(path)
    kind = _suffix(path)
    if kind in _PARQUET:
        return pl.scan_parquet(path)
    if kind in _TEXT:
        return _scan_text(path, _TEXT[kind])
    if kind == ".rds":
        from .rds import ensure_parquet

        return pl.scan_parquet(ensure_parquet(path))
    raise InputError(
        f"Unsupported file type {kind or '(none)'!r} for {path}. "
        "Use .parquet, .csv, .tsv or .rds."
    )


def read_rds(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read a data.frame/data.table saved with ``saveRDS`` (loads it completely)."""
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InputError(
            "Reading .rds files needs pyreadr: pip install 'TRACE[rds]'. "
            "Alternatively convert the file to Parquet in R."
        ) from exc
    path = Path(path)
    log.info("Reading %s completely into memory (RDS cannot be read lazily)", path.name)
    try:
        result = pyreadr.read_r(str(path))
    except Exception as exc:  # pyreadr raises its own error types
        raise InputError(f"Could not read {path} as an R data.frame: {exc}") from exc
    if len(result) != 1:
        raise InputError(f"{path} contains {len(result)} objects; expected one data.frame.")
    frame = result.popitem(last=False)[1]
    if columns is not None:
        _require(frame.columns, columns, path)
        frame = frame[list(columns)]  # drop unused columns before copying to polars
    return pl.from_pandas(frame)


def read_counts(path: str | Path, count_columns: Sequence[str]) -> pl.LazyFrame:
    """Open a per-position count table, keeping only the columns needed.

    Returns a LazyFrame with ``chr`` (str), ``gencoor`` (int), ``cov``,
    ``refSeq`` (str) and the requested count columns (float). ``deletion`` is
    read from the ``-`` column.
    """
    path = Path(path)
    source = {c: (DELETION_SOURCE_COLUMN if c == "deletion" else c) for c in count_columns}
    needed = [*BASE_COLUMNS, *source.values()]
    _check_exists(path)
    kind = _suffix(path)
    if kind in _TEXT and path.stat().st_size > TEXT_TO_PARQUET_BYTES:
        # Large text tables are converted to Parquet once: the data are then read
        # in batches instead of being parsed again (or held in memory) each time.
        from .rds import ensure_parquet_from_text

        lf = pl.scan_parquet(ensure_parquet_from_text(path, _TEXT[kind]))
    else:
        lf = scan_table(path)
    _require(lf.collect_schema().names(), needed, path)
    return lf.select(
        pl.col("chr").cast(pl.String),
        pl.col("gencoor").cast(pl.Int64),
        pl.col("cov").cast(pl.Float64),
        normalise_bases(pl.col("refSeq")).alias("refSeq"),
        *(pl.col(src).cast(pl.Float64).alias(dst) for dst, src in source.items()),
    )


def normalise_bases(col: pl.Expr) -> pl.Expr:
    """Reference bases in upper case, with U written as T (a, u -> A, T)."""
    return col.cast(pl.String).str.to_uppercase().str.replace_all("U", "T")


def read_positions(path: str | Path, value_columns: Sequence[str] = ()) -> pl.LazyFrame:
    """Read a table of positions keyed by ``pos`` ("chr_gencoor") or ``chr`` + ``gencoor``.

    Used for the SNP list. Returns ``chr``,
    ``gencoor`` and ``value_columns``, one row per position.
    """
    path = Path(path)
    lf = scan_table(path)
    names = lf.collect_schema().names()
    if "chr" in names and "gencoor" in names:
        key = [pl.col("chr").cast(pl.String), pl.col("gencoor").cast(pl.Int64)]
    elif "pos" in names:
        # Chromosome/transcript names may contain "_", so split at the last one.
        parts = pl.col("pos").cast(pl.String).str.extract_groups(r"^(.*)_(-?\d+)$")
        key = [
            parts.struct.field("1").alias("chr"),
            parts.struct.field("2").cast(pl.Int64).alias("gencoor"),
        ]
    else:
        raise InputError(
            f"{path.name} needs either a 'pos' column (chr_gencoor) or "
            f"'chr' and 'gencoor' columns. Columns found: {names}"
        )
    _require(names, value_columns, path)
    out = lf.select(*key, *(pl.col(c) for c in value_columns))
    bad = out.filter(pl.col("chr").is_null() | pl.col("gencoor").is_null()).head(3).collect()
    if bad.height:
        raise InputError(
            f"{path.name}: some 'pos' values are not of the form chr_position, "
            f"e.g. rows {bad.to_dicts()}"
        )
    return out.unique(subset=["chr", "gencoor"], keep="first")


def _r_style(df: pl.DataFrame) -> pl.DataFrame:
    """Format values the way R's ``write.csv`` does.

    Whole numbers without decimals (``3``, not ``3.0``), ``Inf``/``-Inf``,
    NaN as ``NA``, and ``TRUE``/``FALSE``.
    """
    exprs = []
    for name, dtype in df.schema.items():
        c = pl.col(name)
        if dtype.is_float():
            exprs.append(
                pl.when(c.is_nan()).then(None)
                .when(c == float("inf")).then(pl.lit("Inf"))
                .when(c == float("-inf")).then(pl.lit("-Inf"))
                .when((c == c.round()) & (c.abs() < 1e15)).then(c.cast(pl.Int64, strict=False).cast(pl.String))
                .otherwise(c.cast(pl.String))
                .alias(name)
            )
        elif dtype == pl.Boolean:
            exprs.append(pl.when(c).then(pl.lit("TRUE")).when(~c).then(pl.lit("FALSE")).alias(name))
    return df.with_columns(exprs) if exprs else df


class TableWriter:
    """Write a table in parts to CSV/TSV (optionally .gz) or Parquet.

    Data go to a temporary file that replaces ``path`` only when writing
    finished without error, so a crash never leaves a truncated result.

        with TableWriter("out.csv") as w:
            for part in parts:
                w.write(part)
    """

    def __init__(self, path: str | Path, *, overwrite: bool = False):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; use overwrite=True / --overwrite to replace it.")
        self.kind = _suffix(self.path)
        if self.kind not in _PARQUET and self.kind not in _TEXT:
            raise InputError(f"Unsupported output type {self.kind!r}; use .csv, .tsv or .parquet.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(f".{self.path.name}.part")
        self._parts: list[Path] = []
        self._handle = None
        self.rows = 0

    def __enter__(self) -> "TableWriter":
        if self.kind in _TEXT:
            if self.path.suffix.lower() == ".gz":
                import gzip

                self._handle = gzip.open(self._tmp, "wb")
            else:
                self._handle = open(self._tmp, "wb")
        return self

    def write(self, df: pl.DataFrame) -> None:
        if self.kind in _TEXT:
            _r_style(df).write_csv(
                self._handle, separator=_TEXT[self.kind], null_value="NA",
                include_header=self.rows == 0 and not self._parts,
            )
            self._parts.append(self._tmp)  # marks header as written
        else:
            part = self._tmp.with_name(f"{self._tmp.name}{len(self._parts)}")
            df.write_parquet(part)
            self._parts.append(part)
        self.rows += df.height

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
            if exc_type is None and self.kind in _PARQUET:
                if len(self._parts) == 1:
                    self._parts[0].replace(self._tmp)
                else:
                    pl.scan_parquet(self._parts).sink_parquet(self._tmp)
            if exc_type is None:
                self._tmp.replace(self.path)
        finally:
            for f in {*self._parts, self._tmp}:
                f.unlink(missing_ok=True)


def write_table(df: pl.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write one table to CSV/TSV (R style, see :func:`_r_style`) or Parquet, by suffix."""
    with TableWriter(path, overwrite=overwrite) as writer:
        writer.write(df)
    return Path(path)
