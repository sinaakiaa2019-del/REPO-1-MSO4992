"""TDA: persistence images -- loading the saved 20x20 image cache."""

import numpy as np
import pandas as pd

from config import OUT


def load_persistence_images():
    """Load the saved 20x20 persistence images and flatten each to 400 pixel columns."""
    with np.load(OUT / "persistence_images.npz") as z:
        images = z["images"].reshape(len(z["images"]), -1)
        dates = pd.to_datetime(z["dates"])
    columns = [f"pi_{i:03}" for i in range(images.shape[1])]
    return pd.DataFrame(images, index=dates, columns=columns)
