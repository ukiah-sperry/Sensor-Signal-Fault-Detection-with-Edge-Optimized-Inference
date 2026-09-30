"""Load CWRU .mat files and segment signals into windows."""
import numpy as np
import scipy.io


def _find_de_key(mat_data: dict) -> str:
    """Return the drive-end accelerometer key from a loaded .mat dict."""
    for key in mat_data:
        if key.endswith("_DE_time"):
            return key
    raise KeyError(f"No _DE_time key found. Available: {list(mat_data.keys())}")


def load_mat_file(path: str) -> np.ndarray:
    """Load a CWRU .mat file and return the drive-end signal as a 1D float32 array."""
    mat = scipy.io.loadmat(path)
    key = _find_de_key(mat)
    signal = mat[key].flatten().astype(np.float32)
    return signal


def segment_signal(signal: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """Split signal into non-overlapping windows. Remainder samples are dropped.

    Returns array of shape (n_windows, window_size).
    """
    n_windows = len(signal) // window_size
    trimmed = signal[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size)


def load_windows_and_labels(
    records: list[dict], window_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Load all records, segment into windows, return (X, y).

    X shape: (total_windows, window_size)
    y shape: (total_windows,) — integer label 0-3
    """
    X_parts, y_parts = [], []
    for rec in records:
        try:
            signal = load_mat_file(rec["path"])
        except Exception as e:
            print(f"Warning: could not load {rec['path']}: {e}")
            continue
        windows = segment_signal(signal, window_size=window_size)
        X_parts.append(windows)
        y_parts.append(np.full(len(windows), rec["label_id"], dtype=np.int64))
    return np.vstack(X_parts), np.concatenate(y_parts)
