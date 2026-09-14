"""TDA: Wasserstein distance between persistence diagrams."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def w2(a, b):
    """Match diagram points, allowing unmatched points to move to the diagonal."""
    n, m = (len(a), len(b))
    c = np.full((n + m, n + m), np.inf)
    c[:n, :m] = cdist(a, b, "sqeuclidean")
    c[np.arange(n), m + np.arange(n)] = np.diff(a).ravel() ** 2 / 2
    c[n + np.arange(m), np.arange(m)] = np.diff(b).ravel() ** 2 / 2
    c[n:, m:] = 0
    i, j = linear_sum_assignment(c)
    return np.sqrt(c[i, j].sum())
