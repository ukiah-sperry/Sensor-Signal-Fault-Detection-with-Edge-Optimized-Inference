"""int8 weight-only quantization and benchmarking for the FaultMLP.

Uses torchao (the PyTorch-recommended modern quantization API) rather than the
deprecated torch.ao.quantization path. Int8WeightOnlyConfig quantizes Linear
layer weights to int8 — no calibration dataset required, CPU-compatible.
"""
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int8WeightOnlyConfig

from src.models.neural import FaultMLP


def quantize_model(model: FaultMLP) -> nn.Module:
    """Return a copy of model with Linear weights quantized to int8.

    Uses torchao Int8WeightOnlyConfig: weights stored as int8, dequantized
    to float32 at inference time. No GPU required.
    """
    q = copy.deepcopy(model)
    q.eval()
    quantize_(q, Int8WeightOnlyConfig())
    return q


def benchmark_model_size(model: nn.Module, save_path: str | Path) -> int:
    """Save model state_dict to save_path, return file size in bytes."""
    save_path = Path(save_path)
    torch.save(model.state_dict(), save_path)
    return save_path.stat().st_size


def benchmark_latency(
    model: nn.Module,
    input_dim: int = 516,
    n_runs: int = 1000,
) -> float:
    """Return mean single-sample inference latency in milliseconds (CPU only).

    10 warm-up runs before timing to avoid cold-start bias.
    """
    model.eval()
    x = torch.randn(1, input_dim)

    with torch.no_grad():
        for _ in range(10):
            model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000  # ms per inference


def run_benchmark(model: FaultMLP, tmp_dir: str | Path = "/tmp") -> dict:
    """Return full benchmark dict: sizes and latencies before/after quantization."""
    tmp = Path(tmp_dir)
    q_model = quantize_model(model)
    input_dim = model.layers[0].in_features

    full_size = benchmark_model_size(model, tmp / "full_precision.pt")
    quant_size = benchmark_model_size(q_model, tmp / "quantized.pt")
    full_latency = benchmark_latency(model, input_dim=input_dim)
    quant_latency = benchmark_latency(q_model, input_dim=input_dim)

    return {
        "full_size_kb": full_size / 1024,
        "quant_size_kb": quant_size / 1024,
        "size_reduction_pct": (1 - quant_size / full_size) * 100,
        "full_latency_ms": full_latency,
        "quant_latency_ms": quant_latency,
        "latency_speedup": full_latency / quant_latency if quant_latency > 0 else float("inf"),
    }


def print_benchmark(results: dict) -> None:
    speedup = results["latency_speedup"]
    latency_note = (
        f"{speedup:.2f}x faster"
        if speedup >= 1.0
        else f"{1/speedup:.2f}x slower (dequantization overhead dominates on small models)"
    )
    print("\n=== Edge Optimization Benchmark (CPU-only, no GPU) ===")
    print(f"{'Metric':<30} {'Full Precision':>15} {'Quantized (int8)':>17}")
    print("-" * 64)
    print(f"{'Model size (KB)':<30} {results['full_size_kb']:>15.1f} {results['quant_size_kb']:>17.1f}")
    print(f"{'Inference latency (ms)':<30} {results['full_latency_ms']:>15.4f} {results['quant_latency_ms']:>17.4f}")
    print("-" * 64)
    print(f"Size reduction:     {results['size_reduction_pct']:.1f}%")
    print(f"Latency change:     {latency_note}")
    print("\nNote: benchmarked on CPU only — simulates edge device constraints.")
    print("No GPU used. No embedded hardware deployment.")
    if speedup < 1.0:
        print("Latency increase is expected for small models: int8 weight-only")
        print("quantization adds dequantization overhead that outweighs compute savings")
        print("at this model size. Size reduction (flash/RAM savings) is the primary benefit.")
