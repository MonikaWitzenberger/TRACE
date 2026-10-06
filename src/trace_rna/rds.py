"""Turn .rds count tables into Parquet automatically, once per file.

Reading a large .rds file from Python (pyreadr) is slow and memory-hungry, while
R itself reads it quickly. So when a count table is given as .rds:

1. If ``<name>.parquet`` already exists next to it and is newer, use that.
2. Otherwise, if R is available, let R read the file and write it as Parquet
   (with the ``arrow`` package) or as CSV (with ``data.table::fwrite``) that is
   then converted to Parquet in streaming mode; both need little memory here.
3. Without R, read it with pyreadr (slow, needs a lot of memory for large
   files) and still save the Parquet copy, so this happens only once.

R is found via the ``TRACE_RSCRIPT`` environment variable, then
``Rscript`` on the PATH, then the standard install folders on Windows.
If the folder of the .rds file is not writable, the Parquet copy goes to a
cache folder in the user's home directory instead.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Converts with arrow if installed; otherwise writes a CSV with data.table::fwrite
# (or base R write.csv) that Python then converts to Parquet in streaming mode.
# Prints the format it wrote ("parquet" or "csv") as the last line.
_R_CONVERT = r"""
args <- commandArgs(trailingOnly = TRUE)
x <- readRDS(args[1])
if (!is.data.frame(x)) { message("not a data.frame: ", paste(class(x), collapse = "/")); quit(status = 4) }
if (requireNamespace("arrow", quietly = TRUE)) {
  arrow::write_parquet(as.data.frame(x), args[2]); cat("parquet\n")
} else if (requireNamespace("data.table", quietly = TRUE)) {
  data.table::fwrite(x, args[3]); cat("csv\n")
} else {
  utils::write.csv(x, args[3], row.names = FALSE); cat("csv\n")
}
"""


def find_rscript() -> str | None:
    """Locate the Rscript executable, or return None."""
    env = os.environ.get("TRACE_RSCRIPT")
    if env:
        return env if Path(env).is_file() else None
    found = shutil.which("Rscript")
    if found:
        return found
    if sys.platform.startswith("win"):
        roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramW6432", "LOCALAPPDATA")]
        candidates: list[str] = []
        for root in filter(None, roots):
            candidates += glob.glob(os.path.join(root, "R", "R-*", "bin", "Rscript.exe"))
            candidates += glob.glob(os.path.join(root, "Programs", "R", "R-*", "bin", "Rscript.exe"))
        if candidates:
            return sorted(candidates)[-1]  # newest version by name
    return None


def parquet_path_for(rds: Path) -> Path:
    """Where the Parquet copy of ``rds`` lives: next to it, or in a user cache folder."""
    beside = rds.with_suffix(".parquet")
    if os.access(rds.parent, os.W_OK):
        return beside
    cache = Path.home() / ".cache" / "trace_rna"
    digest = hashlib.sha1(str(rds.resolve()).encode()).hexdigest()[:10]
    return cache / f"{rds.stem}_{digest}.parquet"


def _is_fresh(parquet: Path, rds: Path) -> bool:
    return parquet.is_file() and parquet.stat().st_mtime >= rds.stat().st_mtime


def _convert_with_r(rscript: str, rds: Path, out: Path) -> bool:
    """Convert with R; True on success, False if R is unavailable or fails.

    Uses arrow when installed; otherwise R writes a CSV (data.table::fwrite or
    write.csv) that is turned into Parquet here without loading it into memory.
    """
    csv_tmp = out.with_name(out.name + ".csv")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(_R_CONVERT)
        script = fh.name
    try:
        proc = subprocess.run(
            [rscript, "--vanilla", script, str(rds), str(out), str(csv_tmp)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log.warning("R could not convert %s: %s", rds.name, (proc.stderr or proc.stdout).strip()[-500:])
            return False
        written = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if written == "csv":
            log.info("R package 'arrow' not installed; converting via CSV instead "
                     "(install.packages(\"arrow\") in R makes this step faster)")
            from .io import _scan_text  # local import: io imports this module

            _scan_text(csv_tmp, ",").sink_parquet(out)
        return out.is_file()
    except OSError as exc:
        log.warning("Could not start R (%s): %s", rscript, exc)
        return False
    finally:
        os.unlink(script)
        csv_tmp.unlink(missing_ok=True)


def ensure_parquet(rds: str | Path, out: str | Path | None = None, *, force: bool = False) -> Path:
    """Return a Parquet copy of ``rds``, creating it if needed (see module docs)."""
    rds = Path(rds)
    target = Path(out) if out is not None else parquet_path_for(rds)
    if not force and _is_fresh(target, rds):
        log.info("Using existing %s", target.name)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.part")
    start = time.monotonic()
    try:
        rscript = find_rscript()
        if rscript:
            log.info("Converting %s to Parquet with R (one time only; about 1-3 minutes)", rds.name)
        if rscript and _convert_with_r(rscript, rds, tmp):
            pass
        else:
            if not rscript:
                log.warning("R not found, so the slow Python reader is used (one time only; can take "
                            "15+ minutes and much memory for large files). Set TRACE_RSCRIPT "
                            "to Rscript's path to use R.")
            from .io import read_rds  # local import: io imports this module

            log.info("Reading %s with Python (one time only; this can take a while)", rds.name)
            read_rds(rds).write_parquet(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("Saved %s (%.1f min); later runs use it directly", target, (time.monotonic() - start) / 60)
    return target


def text_parquet_path(path: Path) -> Path:
    """Parquet cache for a text table: ``x.csv(.gz)`` -> ``x.parquet`` (or the user cache folder)."""
    name = path.name
    for suffix in (".gz",):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    stem = name.rsplit(".", 1)[0]
    beside = path.with_name(stem + ".parquet")
    if os.access(path.parent, os.W_OK):
        return beside
    cache = Path.home() / ".cache" / "trace_rna"
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    return cache / f"{stem}_{digest}.parquet"


def ensure_parquet_from_text(path: str | Path, separator: str) -> Path:
    """Parquet copy of a large CSV/TSV count table (optionally .gz), made once in streaming mode.

    Compressed files are first unpacked to a temporary file, because compressed
    text cannot be read in pieces; neither step loads the table into memory.
    """
    from .io import _scan_text  # local import: io imports this module

    path = Path(path)
    target = text_parquet_path(path)
    if _is_fresh(target, path):
        log.info("Using existing %s", target.name)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.part")
    plain_tmp: Path | None = None
    start = time.monotonic()
    log.info("Converting %s to Parquet (one time only)", path.name)
    try:
        source = path
        if path.name.lower().endswith(".gz"):
            import gzip

            plain_tmp = target.with_name(f".{target.name}.unpacked.csv")
            with gzip.open(path, "rb") as fin, open(plain_tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            source = plain_tmp
        _scan_text(source, separator).sink_parquet(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
        if plain_tmp is not None:
            plain_tmp.unlink(missing_ok=True)
    log.info("Saved %s (%.1f min); later runs use it directly", target, (time.monotonic() - start) / 60)
    return target
