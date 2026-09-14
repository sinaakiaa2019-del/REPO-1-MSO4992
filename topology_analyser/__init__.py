"""TDA: daily persistence diagrams, Betti curves and compact features.

Split into tda_config (epsilon grid), wasserstein (diagram distance),
betti_curves (landscape and network summaries), persistence_images
(saved-image loading) and persistence (the ripser pipeline itself).
Everything is re-exported here so external code is unaffected by the split.
"""

from .tda_config import GRID
from .wasserstein import w2
from .betti_curves import landscape_top1, network_summary
from .persistence_images import load_persistence_images
from .persistence import (
    checksum,
    features,
    compute,
    research_window,
    research_topology,
    run_topology,
)

__all__ = [
    "GRID",
    "w2",
    "landscape_top1",
    "network_summary",
    "load_persistence_images",
    "checksum",
    "features",
    "compute",
    "research_window",
    "research_topology",
    "run_topology",
]
