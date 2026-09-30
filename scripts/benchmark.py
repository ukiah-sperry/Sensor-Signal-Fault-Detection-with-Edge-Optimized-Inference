"""Load trained MLP, apply int8 quantization, print before/after benchmark table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.neural import FaultMLP
from src.optimization.quantize import print_benchmark, quantize_model, run_benchmark


def main():
    model_path = Path("models/mlp_full.pt")
    if not model_path.exists():
        print("ERROR: models/mlp_full.pt not found. Run scripts/train.py first.")
        sys.exit(1)

    print("Loading full-precision MLP...")
    model = FaultMLP(input_dim=516, hidden_dim=64, n_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    print("Running quantization and benchmarks (CPU only, 1000 inference runs)...")
    results = run_benchmark(model, tmp_dir="models")
    print_benchmark(results)

    q_model = quantize_model(model)
    torch.save(q_model.state_dict(), "models/mlp_quantized.pt")
    print("\nSaved quantized model to models/mlp_quantized.pt")


if __name__ == "__main__":
    main()
