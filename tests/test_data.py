
from src.data import make_regression_data


def test_make_regression_data_shapes():
    X, y = make_regression_data(seed=1, n_samples=100, n_features=10, noise=1.0)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
