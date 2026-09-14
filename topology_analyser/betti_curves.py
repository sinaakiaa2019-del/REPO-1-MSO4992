"""TDA: Betti-curve-style summaries -- persistence landscapes and network structure."""

import numpy as np
from scipy.linalg import eigh

from .tda_config import GRID


def landscape_top1(h1, grid=GRID):
    """Top (k=1) persistence landscape: the largest birth/death tent value at each grid point.

    A single landscape function is a more stable summary of a whole diagram
    than one fixed-epsilon Betti coordinate (Bubenik, 2015): every diagram
    point contributes a tent function max(0, min(t-birth, death-t)), and the
    landscape at each t is the largest of those tents, so a small perturbation
    of one point's birth/death moves the landscape only a little, everywhere.
    """
    if len(h1) == 0:
        return np.zeros_like(grid)
    birth = h1[:, 0:1]
    death = h1[:, 1:2]
    tent = np.maximum(0, np.minimum(grid[None, :] - birth, death - grid[None, :]))
    return tent.max(axis=0)


def network_summary(c):
    """Two complementary graph-theoretic summaries of the correlation network.

    Distinct from persistent homology: these describe local redundancy and
    global connectivity of the same weighted graph, not its loop structure.
    Edge weights are max(correlation, 0); negative correlation is not treated
    as a connection. Both are standard, well-established quantities:

    - clustering: a custom weighted closed-walk ratio -- the ratio of
      weighted triangles to weighted length-2 paths, trace(W^3) / off-diag
      sum(W^2). High values mean correlated assets tend to be mutually
      correlated too (redundant, "clumpy" risk); it says nothing about loops.
    - algebraic_connectivity: second-smallest eigenvalue of the normalised
      graph Laplacian (Fiedler value, bounded in [0, 2] regardless of the
      number of eligible assets that day). Low values mean the correlation
      network is close to splitting into weakly-linked blocks.
    """
    n = len(c)
    if n < 2:
        return 0.0, 0.0
    w = np.clip(c, 0, None)
    np.fill_diagonal(w, 0)
    w2_ = w @ w
    denom = w2_.sum() - np.trace(w2_)
    # trace(W^3) = sum_ij (W^2)_ij W_ji; avoid a second cubic product.
    clustering = float(np.sum(w2_ * w.T) / denom) if denom > 0 else 0.0
    degree = w.sum(axis=1)
    inv_sqrt_degree = 1 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian_norm = np.eye(n) - (inv_sqrt_degree[:, None] * w * inv_sqrt_degree[None, :])
    laplacian_norm[degree == 0, degree == 0] = 0.0
    algebraic_connectivity = float(
        eigh(laplacian_norm, subset_by_index=[1, 1], eigvals_only=True)[0]
    )
    return clustering, algebraic_connectivity
