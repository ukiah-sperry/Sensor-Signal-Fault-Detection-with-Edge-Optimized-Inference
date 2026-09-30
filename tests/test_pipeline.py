import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from src.data.loader import segment_signal, load_mat_file
from src.models.classifier import train_classifier, evaluate_classifier


def test_segment_signal_shape():
    """1024-point signal with window=256 → 4 windows."""
    signal = np.arange(1024, dtype=float)
    windows = segment_signal(signal, window_size=256)
    assert windows.shape == (4, 256)


def test_segment_signal_drops_remainder():
    """1030-point signal with window=256 → 4 windows (6 remainder samples dropped)."""
    signal = np.arange(1030, dtype=float)
    windows = segment_signal(signal, window_size=256)
    assert windows.shape == (4, 256)


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


def test_classifier_returns_trained_model(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_classifier(X, y)
    assert hasattr(model, "predict")


def test_classifier_predicts_correct_shape(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_classifier(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_evaluate_returns_confusion_matrix(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_classifier(X, y)
    results = evaluate_classifier(model, X, y)
    assert "confusion_matrix" in results
    cm = results["confusion_matrix"]
    assert cm.shape == (4, 4)


def test_evaluate_accuracy_in_range(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_classifier(X, y)
    results = evaluate_classifier(model, X, y)
    assert 0.0 <= results["accuracy"] <= 1.0


def test_evaluate_per_class_recall(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_classifier(X, y)
    results = evaluate_classifier(model, X, y)
    assert "per_class_recall" in results
    assert len(results["per_class_recall"]) == 4


# --- PyTorch MLP tests ---

import torch
from src.models.neural import FaultMLP, train_mlp, predict_mlp


def test_mlp_forward_shape():
    """MLP output shape matches n_classes."""
    model = FaultMLP(input_dim=516, hidden_dim=64, n_classes=4)
    x = torch.randn(8, 516)
    out = model(x)
    assert out.shape == (8, 4)


def test_mlp_trains_without_error(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_mlp(X, y, epochs=5, lr=1e-3)
    assert isinstance(model, FaultMLP)


def test_mlp_predict_returns_labels(synthetic_dataset):
    X, y = synthetic_dataset
    model = train_mlp(X, y, epochs=5, lr=1e-3)
    preds = predict_mlp(model, X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1, 2, 3})
