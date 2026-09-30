"""End-to-end training: CWRU data → features → RF + MLP → evaluation."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.download import download_dataset
from src.data.loader import load_windows_and_labels
from src.features.extract import extract_features_batch
from src.models.classifier import evaluate_classifier, print_evaluation, train_classifier
from src.models.neural import predict_mlp, train_mlp


def main():
    print("=== Step 1: Loading data ===")
    records = download_dataset("data")

    print("=== Step 2: Segmenting signals ===")
    X_raw, y = load_windows_and_labels(records, window_size=1024)
    print(f"Loaded {len(X_raw)} windows across {len(np.unique(y))} classes")

    print("=== Step 3: Extracting features (FFT + time-domain) ===")
    X = extract_features_batch(X_raw, fs=12000)
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n=== Step 4: Training RandomForest ===")
    rf = train_classifier(X_train, y_train)
    print("--- Train evaluation ---")
    print_evaluation(evaluate_classifier(rf, X_train, y_train))
    print("--- Test evaluation ---")
    print_evaluation(evaluate_classifier(rf, X_test, y_test))

    Path("models").mkdir(exist_ok=True)
    with open("models/rf_classifier.pkl", "wb") as f:
        pickle.dump(rf, f)
    print("Saved models/rf_classifier.pkl")

    print("\n=== Step 5: Training PyTorch MLP ===")
    mlp = train_mlp(X_train, y_train, epochs=30)
    mlp_preds_test = predict_mlp(mlp, X_test)
    mlp_acc = np.mean(mlp_preds_test == y_test)
    print(f"MLP test accuracy: {mlp_acc:.4f}")

    torch.save(mlp.state_dict(), "models/mlp_full.pt")
    print("Saved models/mlp_full.pt")

    print("\nTraining complete. Run scripts/benchmark.py for quantization results.")


if __name__ == "__main__":
    main()
