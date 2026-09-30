import numpy as np
import pytest


@pytest.fixture
def synthetic_dataset():
    """4-class separable dataset: 40 samples × 516 features."""
    rng = np.random.default_rng(42)
    n_per_class = 10
    X = rng.standard_normal((n_per_class * 4, 516)).astype(np.float32)
    for i in range(4):
        X[i * n_per_class: (i + 1) * n_per_class] += i * 5.0
    y = np.repeat(np.arange(4), n_per_class)
    return X, y
