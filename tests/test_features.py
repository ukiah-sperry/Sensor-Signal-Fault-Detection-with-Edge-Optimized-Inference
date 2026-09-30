import numpy as np
import pytest
from src.features.extract import extract_fft_magnitudes, extract_time_features, extract_features, extract_features_batch


def test_fft_peak_at_known_frequency():
    """Pure 100 Hz sine wave → FFT peak within 5 Hz of 100 Hz. Verifiable by math."""
    fs = 1000   # 1 kHz sampling
    n = 1024
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)

    magnitudes = extract_fft_magnitudes(signal, fs=fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    peak_idx = np.argmax(magnitudes)
    peak_freq = freqs[peak_idx]

    assert abs(peak_freq - 100.0) < 5.0, f"Expected peak near 100 Hz, got {peak_freq:.1f} Hz"


def test_fft_two_component_signal():
    """Signal with 50 Hz + 200 Hz components → peaks near both. Verifiable by math."""
    fs = 1000
    n = 1024
    t = np.arange(n) / fs
    signal = (np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    magnitudes = extract_fft_magnitudes(signal, fs=fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    sorted_idx = np.argsort(magnitudes)[::-1]
    top2_freqs = sorted(freqs[sorted_idx[:2]])

    assert abs(top2_freqs[0] - 50.0) < 5.0, f"Expected ~50 Hz, got {top2_freqs[0]:.1f} Hz"
    assert abs(top2_freqs[1] - 200.0) < 5.0, f"Expected ~200 Hz, got {top2_freqs[1]:.1f} Hz"


def test_fft_dc_signal_peak_at_zero():
    """Constant signal → peak at 0 Hz (DC bin). Verifiable by math."""
    signal = np.ones(1024, dtype=np.float32) * 5.0
    magnitudes = extract_fft_magnitudes(signal, fs=1000)
    assert np.argmax(magnitudes) == 0


def test_rms_known_value():
    """RMS of a unit sine wave = 1/sqrt(2). Verifiable analytically."""
    fs = 1000
    t = np.arange(1024) / fs
    signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    feats = extract_time_features(signal)
    expected_rms = 1.0 / np.sqrt(2)
    assert abs(feats["rms"] - expected_rms) < 0.01


def test_peak_to_peak_known_value():
    """Pure sine with amplitude A → peak-to-peak = 2A. Verifiable analytically."""
    t = np.linspace(0, 1, 1024)
    signal = (3.0 * np.sin(2 * np.pi * 5 * t)).astype(np.float32)
    feats = extract_time_features(signal)
    assert abs(feats["peak_to_peak"] - 6.0) < 0.05


def test_extract_features_output_shape():
    """Feature vector for 1024-sample window is 516-dimensional."""
    signal = np.random.randn(1024).astype(np.float32)
    features = extract_features(signal, fs=12000)
    # rfft of 1024 → 513 bins + rms + kurtosis + peak_to_peak = 516
    assert features.shape == (516,)


def test_extract_features_2d_batch():
    """extract_features_batch on (N, 1024) array → (N, 516)."""
    windows = np.random.randn(10, 1024).astype(np.float32)
    feats = extract_features_batch(windows, fs=12000)
    assert feats.shape == (10, 516)
