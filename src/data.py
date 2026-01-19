from __future__ import annotations
import numpy as np


def make_regression_data(
    seed: int, n_samples: int, n_features: int, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))
    y = X @ w + rng.normal(scale=noise, size=(n_samples,))
    return X, y
