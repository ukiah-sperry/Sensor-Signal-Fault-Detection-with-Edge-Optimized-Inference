"""Extract frequency-domain and time-domain features from vibration signal windows."""
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


def extract_fft_magnitudes(signal: np.ndarray, fs: float = 12000) -> np.ndarray:
    """Return FFT magnitude spectrum at positive frequencies.

    For a window of length N, returns N//2 + 1 values (the rfft bins).
    Magnitudes are normalized by N so amplitude is independent of window size.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / n
    # Double non-DC/Nyquist bins to recover true one-sided amplitude
    magnitudes[1:-1] *= 2
    return magnitudes.astype(np.float32)


def extract_time_features(signal: np.ndarray) -> dict[str, float]:
    """Return RMS, kurtosis, and peak-to-peak of a signal window."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    kurt = float(scipy_kurtosis(signal, fisher=True))  # excess kurtosis
    p2p = float(np.max(signal) - np.min(signal))
    return {"rms": rms, "kurtosis": kurt, "peak_to_peak": p2p}


def extract_features(signal: np.ndarray, fs: float = 12000) -> np.ndarray:
    """Extract full feature vector from a single signal window.

    Returns a 1D array of length (n//2 + 1) + 3 = 516 for n=1024.
    Layout: [fft_magnitudes (513), rms (1), kurtosis (1), peak_to_peak (1)]
    """
    fft_feats = extract_fft_magnitudes(signal, fs=fs)
    time_feats = extract_time_features(signal)
    return np.concatenate([
        fft_feats,
        [time_feats["rms"], time_feats["kurtosis"], time_feats["peak_to_peak"]],
    ]).astype(np.float32)


def extract_features_batch(windows: np.ndarray, fs: float = 12000) -> np.ndarray:
    """Extract features from a batch of windows.

    Args:
        windows: array of shape (n_windows, window_size)
        fs: sampling frequency in Hz

    Returns:
        array of shape (n_windows, n_features)
    """
    return np.stack([extract_features(w, fs=fs) for w in windows])
