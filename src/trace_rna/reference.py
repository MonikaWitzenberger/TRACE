"""Original (unconverted) reference sequences from a FASTA file.

Methods on converted references (m5C bisulfite, m6A) need the original base at
each position, because in the aligned data (``refSeq``) an original C already
reads as T (or an original A as G). This module reads the original transcriptome
FASTA, the one that was converted to build the alignment index, and provides
the base per position on demand.

The sequences are kept as bytes (about 1 byte per base); the per-position table
(``chr``, ``gencoor``, ``base``) is built only for the transcripts that are
being processed, so even a full transcriptome needs little memory.

FASTA details handled:

* the header up to the first whitespace is the transcript name
  (``>uc001aaa.3 some description`` -> ``uc001aaa.3``); it must match ``chr``
  in the count tables;
* lower- and upper-case bases, U read as T, Windows (CRLF) line endings,
  sequences on one or many lines, ``.gz`` compression;
* positions are 1-based, like ``gencoor``.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fas")

_UPPER = bytes.maketrans(b"acgtunU", b"ACGTTNT")
_BASES = np.array(list("ACGTN"))
_CODE = np.full(256, 4, dtype=np.uint8)  # anything unexpected -> N
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i


def is_fasta(path: str | Path) -> bool:
    """True if the file name ends in a FASTA suffix (optionally followed by .gz)."""
    name = Path(path).name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return name.endswith(FASTA_SUFFIXES)


class ReferenceFasta:
    """Original reference sequences, looked up by transcript name."""

    def __init__(self, sequences: dict[str, bytes], source: str = "<memory>"):
        if not sequences:
            raise ValueError(f"No sequences found in {source}")
        self.sequences = sequences
        self.source = source

    @classmethod
    def read(cls, path: str | Path) -> "ReferenceFasta":
        path = Path(path)
        if not path.is_file():
            from .io import InputError

            raise InputError(f"File not found: {path}")
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        sequences: dict[str, bytes] = {}
        name: str | None = None
        parts: list[bytes] = []
        with opener(path, "rb") as fh:
            for line in fh:
                line = line.rstrip(b"\r\n")
                if line.startswith(b">"):
                    if name is not None:
                        cls._store(sequences, name, parts, path)
                    fields = line[1:].split()
                    name = fields[0].decode() if fields else ""
                    parts = []
                elif line and name is not None:
                    parts.append(line.strip())
            if name is not None:
                cls._store(sequences, name, parts, path)
        log.info("Read %d sequences (%d bases) from %s", len(sequences),
                 sum(map(len, sequences.values())), path.name)
        return cls(sequences, path.name)

    @staticmethod
    def _store(sequences: dict[str, bytes], name: str, parts: list[bytes], path: Path) -> None:
        if not name:
            raise ValueError(f"{path.name}: a FASTA header has no name")
        if name in sequences:
            raise ValueError(f"{path.name}: sequence name {name!r} occurs more than once")
        sequences[name] = b"".join(parts).translate(_UPPER)

    @property
    def names(self) -> list[str]:
        return list(self.sequences)

    def lengths(self) -> pl.DataFrame:
        """One row per sequence: ``chr`` and its length."""
        return pl.DataFrame({
            "chr": list(self.sequences),
            "length": [len(s) for s in self.sequences.values()],
        }, schema={"chr": pl.String, "length": pl.Int64})

    def table(self, chroms: Iterable[str] | None = None) -> pl.DataFrame:
        """Per-position table ``chr``, ``gencoor`` (1-based), ``base`` for ``chroms``.

        Names not in the FASTA are skipped. ``chroms=None`` means all sequences.
        """
        names = [c for c in (self.sequences if chroms is None else dict.fromkeys(chroms))
                 if c in self.sequences]
        if not names:
            return pl.DataFrame(schema={"chr": pl.String, "gencoor": pl.Int64, "base": pl.String})
        seqs = [self.sequences[c] for c in names]
        lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        codes = _CODE[np.frombuffer(b"".join(seqs), dtype=np.uint8)]
        starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        gencoor = np.arange(int(lengths.sum()), dtype=np.int64) - starts + 1
        chr_col = pl.Series("chr", names, dtype=pl.String).gather(
            np.repeat(np.arange(len(names), dtype=np.int64), lengths)
        )
        base = pl.Series("base", _BASES).gather(codes)
        return pl.DataFrame([chr_col, pl.Series("gencoor", gencoor), base])

    def lazy_table(self, chroms: Sequence[str] | None) -> pl.LazyFrame:
        return self.table(chroms).lazy()
