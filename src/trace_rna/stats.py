"""Vectorised statistics that reproduce R's behaviour exactly."""

from __future__ import annotations

import numpy as np
from scipy.special import chdtrc


def chisq_2x2_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    min_total: float = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """P-values of R's ``chisq.test(matrix(c(a, c, b, d), 2))`` for many 2x2 tables.

    Each table is ``[[a, b], [c, d]]`` (rows = samples, columns = unmodified /
    modified counts). Pearson's chi-square with Yates' continuity correction,
    exactly as R computes it (``correct = TRUE``, the default):

    * ``YATES = min(0.5, min |O - E|)`` over the four cells,
    * ``X2 = sum((|O - E| - YATES)^2 / E)``, 1 degree of freedom.

    Returns ``(pvalues, is_na)``. ``is_na`` marks tables with a missing count
    or a total below ``min_total``; the original scripts return ``NA`` there
    (the p-value array holds NaN at those rows). Tables with a zero expected
    count give NaN with ``is_na`` False, which is what R's ``chisq.test``
    returns, so ``NA`` and ``NaN`` stay distinguishable in the output as in R.
    """
    x = np.stack([a, b, c, d], axis=1).astype(np.float64)  # (n, 4)
    n = x.sum(axis=1)
    row1, row2 = x[:, 0] + x[:, 1], x[:, 2] + x[:, 3]
    col1, col2 = x[:, 0] + x[:, 2], x[:, 1] + x[:, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.stack(
            [row1 * col1, row1 * col2, row2 * col1, row2 * col2], axis=1
        ) / n[:, None]
        dev = np.abs(x - expected)
        yates = np.minimum(0.5, dev.min(axis=1))
        stat = (((dev - yates[:, None]) ** 2) / expected).sum(axis=1)
        pval = chdtrc(1.0, stat)

    invalid = np.isnan(x).any(axis=1) | (n < min_total)
    pval[invalid] = np.nan
    return pval, invalid
