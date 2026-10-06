"""TRACE: RNA modification site index and gene index from per-position count tables."""

__version__ = "1.0.0"

from .methods import Method, available_methods, get_method, register_method  # noqa: E402
from .geneindex import compute_geneindex, iter_geneindex, write_geneindex  # noqa: E402
from .common import Resources  # noqa: E402
from .siteindex import compute_siteindex, iter_siteindex, write_siteindex  # noqa: E402

__all__ = [
    "Method",
    "Resources",
    "compute_geneindex",
    "available_methods",
    "compute_siteindex",
    "get_method",
    "iter_geneindex",
    "iter_siteindex",
    "register_method",
    "write_geneindex",
    "write_siteindex",
]
