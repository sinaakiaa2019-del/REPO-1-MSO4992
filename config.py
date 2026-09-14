"""Shared paths and runtime setup; experiment settings are in config.json."""

from pathlib import Path
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if os.environ.get("TDA_DEPS"):
    sys.path.insert(0, os.environ["TDA_DEPS"])
ROOT = Path(__file__).resolve().parent
# data_pipeline/ and ml/ hold Octavian's and Sina's modules; adding them here
# means every existing `from matrix_builder import ...`-style import keeps
# resolving unchanged no matter which file it is written in.
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "ml"))
HERE = ROOT / "cache"
OUT = HERE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib")
