"""Train and evaluate a RandomForest classifier for bearing fault detection."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score


CLASS_NAMES = ["normal", "inner", "ball", "outer"]


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a RandomForest on feature matrix X with integer labels y."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Return accuracy, per-class recall, and confusion matrix.

    Per-class recall matters more than overall accuracy in fault detection:
    missing a real fault (false negative) is worse than a false alarm.
    """
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds)
    per_class_recall = recall_score(y, preds, average=None, zero_division=0)
    return {
        "accuracy": float(acc),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm,
        "predictions": preds,
    }


def print_evaluation(results: dict) -> None:
    """Print a human-readable evaluation summary."""
    print(f"\nOverall accuracy: {results['accuracy']:.4f}")
    print("\nPer-class recall (fraction of true faults detected):")
    for name, recall in zip(CLASS_NAMES, results["per_class_recall"]):
        print(f"  {name:8s}: {recall:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"  {'':8s}", "  ".join(f"{n:6s}" for n in CLASS_NAMES))
    for i, row in enumerate(results["confusion_matrix"]):
        print(f"  {CLASS_NAMES[i]:8s}", "  ".join(f"{v:6d}" for v in row))
