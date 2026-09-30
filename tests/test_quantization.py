import numpy as np
import torch
import torch.nn as nn
import pytest
from src.models.neural import FaultMLP, train_mlp, predict_mlp
from src.optimization.quantize import quantize_model, benchmark_model_size, benchmark_latency


@pytest.fixture
def trained_mlp():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 516)).astype(np.float32)
    for i in range(4):
        X[i * 20: (i + 1) * 20] += i * 5.0
    y = np.repeat(np.arange(4), 20)
    return train_mlp(X, y, epochs=10)


def test_quantize_returns_module(trained_mlp):
    q = quantize_model(trained_mlp)
    assert isinstance(q, nn.Module)


def test_quantized_model_smaller(trained_mlp, tmp_path):
    q = quantize_model(trained_mlp)
    full_size = benchmark_model_size(trained_mlp, tmp_path / "full.pt")
    quant_size = benchmark_model_size(q, tmp_path / "quant.pt")
    assert quant_size < full_size, (
        f"Expected quantized ({quant_size}B) < full ({full_size}B)"
    )


def test_quantized_model_predicts_same_shape(trained_mlp):
    """Quantized model must still produce valid class predictions."""
    q = quantize_model(trained_mlp)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 516)).astype(np.float32)
    preds_full = predict_mlp(trained_mlp, X)
    preds_quant = predict_mlp(q, X)
    assert preds_quant.shape == preds_full.shape
    assert set(preds_quant).issubset({0, 1, 2, 3})


def test_quantized_accuracy_not_destroyed(trained_mlp):
    """Quantized model keeps ≥70% agreement with full-precision model on same inputs."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 516)).astype(np.float32)
    for i in range(4):
        X[i * 25: (i + 1) * 25] += i * 5.0

    q = quantize_model(trained_mlp)
    preds_full = predict_mlp(trained_mlp, X)
    preds_quant = predict_mlp(q, X)
    agreement = np.mean(preds_full == preds_quant)
    assert agreement >= 0.70, f"Quantized model deviates too much: {agreement:.2%} agreement"


def test_benchmark_latency_returns_float(trained_mlp):
    ms = benchmark_latency(trained_mlp, input_dim=516, n_runs=50)
    assert isinstance(ms, float)
    assert ms > 0
