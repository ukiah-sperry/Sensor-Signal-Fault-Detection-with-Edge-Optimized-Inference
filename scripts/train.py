"""End-to-end training: CWRU data → features → RF + MLP → evaluation."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.download import download_dataset
from src.data.loader import load_windows_and_labels
from src.features.extract import extract_features_batch
from src.models.classifier import evaluate_classifier, print_evaluation, train_classifier
from src.models.neural import predict_mlp, train_mlp

# Hold out the 2HP recording for each class as the test set.
# CWRU_FILES ordering: each class has 3 files (0HP, 1HP, 2HP) at sequential indices.
# file_id → class: 0-2 normal, 3-5 inner, 6-8 ball, 9-11 outer.
# 2HP files are at indices 2, 5, 8, 11 — slightly different RPM from train files,
# so this tests generalization across load conditions, not just held-out windows
# from the same recording.
TEST_FILE_IDS = {2, 5, 8, 11}


def main():
    print("=== Step 1: Loading data ===")
    records = download_dataset("data")

    print("=== Step 2: Segmenting signals ===")
    X_raw, y, file_ids = load_windows_and_labels(records, window_size=1024)
    print(f"Loaded {len(X_raw)} windows across {len(np.unique(y))} classes")

    print("=== Step 3: Extracting features (FFT + time-domain) ===")
    X = extract_features_batch(X_raw, fs=12000)
    print(f"Feature matrix: {X.shape}")

    # Grouped split: all windows from a given recording stay on the same side.
    # Splitting by individual windows would leak: adjacent windows from the same
    # recording appear in both train and test, inflating accuracy artificially.
    test_mask = np.isin(file_ids, list(TEST_FILE_IDS))
    train_mask = ~test_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_files = sorted(set(file_ids[train_mask].tolist()))
    test_files = sorted(set(file_ids[test_mask].tolist()))
    print(f"Train: {len(X_train)} windows from files {train_files}")
    print(f"Test:  {len(X_test)} windows from files {test_files} (2HP recordings — different RPM)")

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
