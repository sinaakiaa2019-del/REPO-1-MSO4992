"""Matrix builder: centred, weighted and stabilised return correlation."""

from config import ROOT, HERE, OUT
import numpy as np


def correlation(a, lam=0.94, shrink=0.1):
    """Centre observed returns, favour recent days and stabilise correlation."""
    assert np.isfinite(a).all()
    w = lam ** np.arange(len(a) - 1, -1, -1)
    w /= w.sum()
    centered = a - w @ a
    z = centered * np.sqrt(w[:, None])
    cov = z.T @ z
    sd = np.sqrt(np.diag(cov))
    assert (sd > 1e-12).all()
    c = np.clip(cov / np.outer(sd, sd), -1, 1)
    c = (1 - shrink) * c + shrink * np.eye(c.shape[0])
    np.fill_diagonal(c, 1)
    return c
