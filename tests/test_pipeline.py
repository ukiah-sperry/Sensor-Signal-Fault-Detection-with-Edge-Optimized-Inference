import numpy as np
import pytest
from src.data.loader import segment_signal, load_mat_file


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
