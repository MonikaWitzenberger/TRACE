"""Command-line interface: ``trace <command> --help`` for details."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Sequence

import polars as pl

from . import __version__
from .common import DEFAULT_CHUNK_ROWS, Resources
from .geneindex import write_geneindex
from .io import InputError, scan_table
from .methods import available_methods
from .rds import ensure_parquet
from .siteindex import write_siteindex

log = logging.getLogger("trace_rna")

LEVELS: dict[str, Callable[..., int]] = {"site": write_siteindex, "gene": write_geneindex}
SITE_COV_DEFAULT = 20  # the gene index has no coverage threshold
SHEET_COLUMNS_REQUIRED = ("treated", "control")
SHEET_COLUMNS_OPTIONAL = ("method", "level", "cov_threshold", "experiment", "output")


def _add_index_options(p: argparse.ArgumentParser, *, level: str | None) -> None:
    in_batch = level is None
    g = p.add_argument_group("index calculation")
    g.add_argument(
        "--method", required=not in_batch,
        help="Modification method, see 'trace methods'"
        + (" (default for rows without a 'method' value)" if in_batch else ""),
    )
    if level != "gene":
        g.add_argument(
            "--cov-threshold", type=float, default=None,
            help=f"Site index only: keep sites with coverage of at least this (default: {SITE_COV_DEFAULT})",
        )

    r = p.add_argument_group("annotation files (paths are never built in)")
    r.add_argument("--snps", type=Path, help="Positions to exclude: column 'pos' (chr_position) or 'chr'+'gencoor'")
    r.add_argument("--reference", type=Path, help="Original (unconverted) transcriptome as FASTA (.fa/.fasta, optionally .gz), needed for "
                        "m5C, m5C_CC and m6A")
    r.add_argument(
        "--always-keep-chr", action="append", default=[], metavar="CHR",
        help="Transcript (e.g. a spike-in such as MS2) that is always kept: site index without the "
             "coverage filter and with sites found in only one sample; gene index summed over all its "
             "sites in each sample, even if only one sample has it (repeatable)",
    )
    p.add_argument(
        "--experiment",
        help="Label written to the 'experiment' column (the file names are always written to "
             "'datafile1' and 'datafile2')" + (" (default for rows without one)" if in_batch else ""),
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    p.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, metavar="N",
        help="Process whole transcripts in batches of about N input rows; lower it if memory "
             "is tight (results are identical; default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace",
        description="RNA modification site and gene index from per-position nucleotide count tables.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Show progress details")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Only show errors")
    sub = parser.add_subparsers(dest="command", required=True)

    for level in LEVELS:
        p = sub.add_parser(f"{level}index",
                           help=f"{level.capitalize()} index for one treated/control pair")
        p.add_argument("--treated", type=Path, required=True, help="Treated sample (.x): .parquet, .csv, .tsv or .rds")
        p.add_argument("--control", type=Path, required=True, help="Control sample (.y)")
        p.add_argument("-o", "--output", type=Path, required=True, help="Result file (.csv, .csv.gz, .tsv or .parquet)")
        _add_index_options(p, level=level)
        p.set_defaults(level=level)

    b = sub.add_parser(
        "batch",
        help="Calculate site or gene index for many pairs listed in a sample sheet",
        description=(
            "Sample sheet (CSV/TSV) columns: treated, control (required); method, level (site/gene), "
            "cov_threshold (site rows only), experiment, output (optional; empty cells use the options). "
            "Relative paths are resolved against the sheet's folder."
        ),
    )
    b.add_argument("sheet", type=Path, help="Sample sheet (.csv or .tsv)")
    b.add_argument("--outdir", type=Path, required=True, help="Folder for results without an 'output' entry")
    b.add_argument("--level", choices=sorted(LEVELS), default="site",
                   help="Index level (site or gene) for rows without a 'level' value (default: %(default)s)")
    b.add_argument("--stop-on-error", action="store_true", help="Stop at the first failing row")
    _add_index_options(b, level=None)

    sub.add_parser("methods", help="List the available modification methods")

    c = sub.add_parser("convert", help="Convert an .rds count table to Parquet now (otherwise done automatically on first use)")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path, nargs="?", help="Default: input name with .parquet")
    c.add_argument("--overwrite", action="store_true")
    return parser


def _resources(args: argparse.Namespace) -> Resources:
    return Resources.load(
        snps=args.snps,
        reference=args.reference,
        always_keep_chr=getattr(args, "always_keep_chr", ()),
    )


def _experiment(args: argparse.Namespace, override: str | None = None) -> str | None:
    return str(override) if override else args.experiment


def _calculate(level: str, treated: Path, control: Path, output: Path, method: str,
           cov_threshold: float | None, resources: Resources, experiment: str | None,
           args: argparse.Namespace) -> int:
    kwargs = dict(overwrite=args.overwrite, experiment=experiment, chunk_rows=args.chunk_rows)
    if level == "site":
        kwargs["cov_threshold"] = SITE_COV_DEFAULT if cov_threshold is None else cov_threshold
    return LEVELS[level](treated, control, method, output, resources=resources, **kwargs)


def _run_single(args: argparse.Namespace) -> int:
    n = _calculate(args.level, args.treated, args.control, args.output, args.method, getattr(args, "cov_threshold", None),
               _resources(args), _experiment(args), args)
    log.info("Wrote %d %ss to %s", n, args.level, args.output)
    return 0


def _cell(row: dict, key: str):
    value = row.get(key)
    return None if value is None or (isinstance(value, str) and not value.strip()) else value


def _run_batch(args: argparse.Namespace) -> int:
    sheet = scan_table(args.sheet).collect()
    missing = [c for c in SHEET_COLUMNS_REQUIRED if c not in sheet.columns]
    if missing and len(sheet.columns) == 1 and ";" in sheet.columns[0]:
        raise InputError(f"{args.sheet.name} uses ';' as separator (as German Excel saves CSV files). "
                         "Please save it as 'CSV UTF-8 (comma delimited)'.")
    if missing:
        raise InputError(f"Sample sheet lacks column(s) {missing}; found {sheet.columns}")
    unknown = set(sheet.columns) - set(SHEET_COLUMNS_REQUIRED) - set(SHEET_COLUMNS_OPTIONAL)
    if unknown:
        log.warning("Ignoring sample sheet column(s): %s", sorted(unknown))

    base = args.sheet.resolve().parent
    resolve = lambda p: p if p.is_absolute() else base / p  # noqa: E731
    resources = _resources(args)  # loaded once for all rows
    failures = 0
    for i, row in enumerate(sheet.iter_rows(named=True), start=1):
        try:
            method = _cell(row, "method") or args.method
            if not method:
                raise ValueError("no method in the sheet row and no --method given")
            level = str(_cell(row, "level") or args.level).strip().lower()
            if level not in LEVELS:
                raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
            treated, control = resolve(Path(row["treated"])), resolve(Path(row["control"]))
            cov = _cell(row, "cov_threshold")
            if level == "gene" and cov is not None:
                log.warning("[%d/%d] cov_threshold is ignored for the gene index (it has no coverage threshold)",
                            i, sheet.height)
            out = _cell(row, "output")
            out_path = (resolve(Path(out)) if out
                        else args.outdir / f"{treated.name.split('.')[0]}_{method}_{level}index.csv")
            log.info("[%d/%d] %s vs %s (%s, %s)", i, sheet.height, treated.name, control.name, method, level)
            n = _calculate(level, treated, control, out_path, method,
                       float(cov) if cov is not None else args.cov_threshold,
                       resources, _experiment(args, _cell(row, "experiment")), args)
            log.info("[%d/%d] wrote %d %ss to %s", i, sheet.height, n, level, out_path)
        except Exception as exc:  # report and continue with the next row
            failures += 1
            log.error("[%d/%d] failed: %s", i, sheet.height, exc)
            if args.stop_on_error:
                break
    log.info("%d of %d rows succeeded", sheet.height - failures, sheet.height)
    return 1 if failures else 0


def _run_methods(_: argparse.Namespace) -> int:
    for name, m in available_methods().items():
        motif = ", ".join(f"{o:+d}:{'/'.join(sorted(b))}" for o, b in sorted(m.motif.items()))
        extra = f" [motif {motif} on {m.motif_column}]" if motif else ""
        ref = f" [needs --reference, original base {m.reference_base}]" if m.reference_base else ""
        print(f"{name:12s} {m.description}{extra}{ref}")
    return 0


def _run_convert(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise InputError(f"File not found: {args.input}")
    out = args.output or args.input.with_suffix(".parquet")
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite to replace it.")
    ensure_parquet(args.input, out, force=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    handlers = {"siteindex": _run_single, "geneindex": _run_single,
                "batch": _run_batch,
                "methods": _run_methods, "convert": _run_convert}
    try:
        return handlers[args.command](args)
    except (InputError, ValueError, FileExistsError) as exc:
        log.error("%s", exc)
    except pl.exceptions.PolarsError as exc:
        detail = next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), "")
        log.error("Could not read the input: %s. Check that the count columns (cov, A, C, G, T, -) "
                  "contain only numbers and 'gencoor' only whole numbers.", detail.rstrip("."))
    except (OSError, EOFError) as exc:  # missing/unreadable/damaged files, no permission, disk full
        name = getattr(exc, "filename", None)
        if name:  # show the output file, not its temporary ".name.part" twin
            p = Path(str(name))
            if p.name.startswith(".") and ".part" in p.name:
                name = p.with_name(p.name[1:].split(".part")[0])
        log.error("File problem%s: %s", f" with {name}" if name else "", getattr(exc, "strerror", None) or exc)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
