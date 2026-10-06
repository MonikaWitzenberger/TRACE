"""Modification methods: which bases to filter on and how to count modified reads.

Each method is a :class:`Method` record. The site index is

    site index = sum(modified) / (sum(modified) + sum(unmodified))

computed separately for the treated (``.x``) and control (``.y``) sample.
New methods can be added with :func:`register_method` without touching the
pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

MotifColumn = Literal["refSeq", "base"]


@dataclass(frozen=True)
class Method:
    """Definition of one modification method (used for site and gene index).

    Attributes:
        name: Identifier used on the command line (e.g. ``"m1A"``).
        description: One line shown by ``trace methods``.
        ref_base: Required value of ``refSeq`` at the site (the base the reads
            were aligned against).
        modified: Count columns that signal the modification (numerator).
        unmodified: Count columns that signal the unmodified base.
        reference_base: For converted-reference data (bisulfite etc.): the
            required *original* base at the site, taken from the reference
            FASTA. ``None`` if no reference is needed.
        motif: Required neighbouring bases, as ``{offset: allowed bases}``;
            offset -1 is the 5' neighbour, +1 the 3' neighbour.
        motif_column: Whether ``motif`` is checked against ``refSeq`` or the
            original base from the reference FASTA.
        control_missing_as_zero: Set the control site index to 0 where it cannot
            be computed (used for m5C).
        report_sum: Also output ``sumNuc.x``/``sumNuc.y`` (sum of the modified
            columns), for multi-base mismatch methods.
    """

    name: str
    description: str
    ref_base: str
    modified: tuple[str, ...]
    unmodified: tuple[str, ...]
    reference_base: str | None = None
    motif: Mapping[int, frozenset[str]] = field(default_factory=dict)
    motif_column: MotifColumn = "refSeq"
    control_missing_as_zero: bool = False
    report_sum: bool = False

    @property
    def needs_reference(self) -> bool:
        return self.reference_base is not None or self.motif_column == "base"

    @property
    def count_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.unmodified + self.modified))


def _motif(**kwargs: str) -> dict[int, frozenset[str]]:
    """Helper: ``_motif(m1="CA", p1="G")`` -> ``{-1: {C, A}, 1: {G}}``."""
    out: dict[int, frozenset[str]] = {}
    for key, bases in kwargs.items():
        sign = -1 if key[0] == "m" else 1
        out[sign * int(key[1:])] = frozenset(bases)
    return out


_REGISTRY: dict[str, Method] = {}


def register_method(method: Method, *, replace: bool = False) -> None:
    """Add a method to the registry so the CLI and API can use it by name."""
    if method.name in _REGISTRY and not replace:
        raise ValueError(f"Method {method.name!r} already exists; pass replace=True.")
    if not method.modified or not method.unmodified:
        raise ValueError("A method needs at least one modified and one unmodified column.")
    if method.motif_column == "base" and method.reference_base is None:
        raise ValueError("motif_column='base' needs reference_base to be set.")
    _REGISTRY[method.name] = method


def get_method(name: str) -> Method:
    if name in _REGISTRY:
        return _REGISTRY[name]
    by_lower = {k.lower(): v for k, v in _REGISTRY.items()}
    if name.lower() in by_lower:  # e.g. "ai" or "M5C"
        return by_lower[name.lower()]
    known = ", ".join(sorted(_REGISTRY))
    raise ValueError(f"Unknown method {name!r}. Available methods: {known}")


def available_methods() -> Mapping[str, Method]:
    return MappingProxyType(_REGISTRY)


for _m in (
    Method("AI", "A-to-I editing: G read at A", "A", ("G",), ("A",)),
    Method("CU", "C-to-U deamination: T read at C", "C", ("T",), ("C",)),
    Method("m1A", "m1A: any mismatch at A", "A", ("G", "C", "T"), ("A",),
           report_sum=True),
    Method("m1G", "m1G: any mismatch at G", "G", ("A", "C", "T"), ("G",),
           report_sum=True),
    Method("ac4C", "ac4C (ac4C-seq): T read at C", "C", ("T",), ("C",)),
    Method("ac4C_CCG", "ac4C (ac4C-seq): T read at C in a CCG motif", "C", ("T",), ("C",),
           motif=_motif(m1="C", p1="G")),
    Method("pseudo", "Pseudouridine (BID-seq): deletion at U", "T", ("deletion",), ("T",)),
    Method("m5C", "m5C (RNA bisulfite-seq): unconverted C at original C", "T", ("C",), ("T",),
           reference_base="C", control_missing_as_zero=True),
    Method("m5C_CC", "m5C (RNA bisulfite-seq): as m5C, original C preceded by C or U", "T",
           ("C",), ("T",), reference_base="C", motif=_motif(m1="CT"),
           motif_column="base", control_missing_as_zero=True),
    Method("m6A", "m6A (GLORI): unconverted A at original A", "G",
           ("A",), ("G",), reference_base="A"),
):
    register_method(_m)
