"""Data/TDA entry point, using the same stages as run_all.py."""

import argparse
from pathlib import Path
from run_all import stage, validate_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    validate_protocol()
    for name in ["data", "math", "tda", "coverage"]:
        stage(name, args)
