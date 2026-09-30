# Sensor Signal Fault Detection with Edge-Optimized Inference

Bearing fault classification from vibration sensor data, with CPU-only model quantization benchmarking. Demonstrates signal processing, ML for sensor data, and edge optimization techniques relevant to semiconductor/industrial-IoT engineering roles.

[![CI](https://github.com/ukiahsperry/sensor-fault-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ukiahsperry/sensor-fault-detection/actions/workflows/ci.yml)

---

## What This Is (and Is Not)

This project demonstrates **edge optimization techniques** — model quantization and CPU-only inference benchmarking — applied to a real industrial sensor dataset. It does **not** include deployment on embedded or edge hardware. All benchmarks were run on a standard development laptop (CPU only, no GPU). "Edge-optimized" means: smaller model size and faster inference under the resource constraints (no GPU, CPU only) that matter on real edge devices.

---

## Dataset

[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) — publicly available, widely cited industrial vibration dataset. Drive-end accelerometer data from rolling-element bearings in four conditions:

| Label | Condition | Count (windows) |
|-------|-----------|----------------|
| 0 | Normal | 1182 |
| 1 | Inner race fault | 356 |
| 2 | Ball fault | 355 |
| 3 | Outer race fault | 356 |

12 .mat files, 3 load conditions per fault class (0 HP, 1 HP, 2 HP). ~2,249 windows total at 1024 samples/window (12 kHz sampling rate, ~85 ms per window).

---

## Pipeline

```
Raw .mat vibration signal (12 kHz, drive-end accelerometer)
        ↓
Window segmentation (1024 samples, non-overlapping)
        ↓
Feature extraction:
  • FFT magnitude spectrum (513 bins, 0–6 kHz)
  • RMS, excess kurtosis, peak-to-peak
  → 516-dimensional feature vector per window
        ↓
      ┌─────────────────────────┐   ┌──────────────────────────────┐
      │  RandomForest (sklearn) │   │  PyTorch MLP (edge target)   │
      │  100 trees, n_jobs=-1   │   │  516 → 64 → 64 → 4 classes  │
      └─────────────────────────┘   └──────────────────────────────┘
                                             ↓
                                  int8 weight-only quantization
                                       (torchao)
                                             ↓
                                  Benchmark: size + latency
                                  before vs. after
```

---

## Results

### Classification (test set, 80/20 split)

| Model | Accuracy | Normal recall | Inner recall | Ball recall | Outer recall |
|-------|----------|--------------|-------------|------------|-------------|
| RandomForest | 100.0% | 100% | 100% | 100% | 100% |
| PyTorch MLP  | 100.0% | 100% | 100% | 100% | 100% |

CWRU bearing data is well-separated in frequency space — fault signatures are distinct peaks in the FFT spectrum, so both models classify cleanly. Per-class recall is reported rather than just overall accuracy because missing a real fault (false negative) is a worse error than a false alarm in predictive maintenance contexts.

### Edge Optimization Benchmark (CPU-only, no GPU)

| Metric | Full Precision | Quantized (int8) |
|--------|---------------|-----------------|
| Model size | 149.6 KB | 42.7 KB |
| Inference latency | 0.0176 ms | 0.0612 ms |
| **Size reduction** | — | **71.4%** |
| **Latency change** | — | **3.5x slower** |

**On size:** 71.4% reduction means the quantized model needs 3.5× less flash/RAM — the primary benefit for memory-constrained edge devices.

**On latency:** The quantized model is slower on this hardware, which is expected for a model this small. With int8 weight-only quantization, weights are stored as int8 and dequantized to float32 at inference time. For a 3-layer, 516→64→64→4 network, the compute is so lightweight that the dequantization overhead outweighs any savings. On larger models or memory-bandwidth-bound hardware (common on real edge devices), int8 quantization typically reduces latency.

Benchmarked: 1000 inference runs, single sample, CPU only. No GPU used. No embedded hardware deployment.

---

## Signal Processing Tests

The preprocessing functions are tested against synthetic signals with **mathematically verifiable expected outputs** — not just "does it run without error":

| Test | Signal | Expected result | Why it's verifiable |
|------|--------|-----------------|---------------------|
| `test_fft_peak_at_known_frequency` | Pure 100 Hz sine, 1 kHz fs | FFT peak within 5 Hz of 100 Hz | DFT formula: bin k = freq × N / fs |
| `test_fft_two_component_signal` | 50 Hz + 200 Hz sine sum | Two largest FFT peaks near 50 Hz and 200 Hz | Superposition + DFT linearity |
| `test_fft_dc_signal_peak_at_zero` | Constant signal | FFT peak at bin 0 (DC) | DC component = bin 0 by definition |
| `test_rms_known_value` | Unit sine wave | RMS = 1/√2 ≈ 0.707 | RMS of sin = A/√2 analytically |
| `test_peak_to_peak_known_value` | Amplitude-3 sine | Peak-to-peak = 6.0 | 2A for any pure sinusoid |

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download CWRU dataset (~30 MB, 12 files)
python scripts/download_data.py

# Train models (RF + MLP)
python scripts/train.py

# Run quantization benchmark
python scripts/benchmark.py

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
src/
  data/
    download.py     — CWRU .mat file fetcher (12-file curated subset)
    loader.py       — .mat → windowed numpy arrays + integer labels
  features/
    extract.py      — FFT magnitude spectrum + RMS + kurtosis + peak-to-peak
  models/
    classifier.py   — RandomForest training, evaluation, confusion matrix
    neural.py       — PyTorch MLP definition, training loop, prediction
  optimization/
    quantize.py     — int8 weight-only quantization (torchao) + benchmarking

scripts/
  download_data.py  — CLI wrapper for dataset download
  train.py          — end-to-end training run
  benchmark.py      — quantization before/after table

tests/
  conftest.py           — shared synthetic dataset fixture
  test_features.py      — 7 math-verifiable signal processing tests
  test_pipeline.py      — RF + MLP training/eval with synthetic fixtures
  test_quantization.py  — quantized model sanity checks
```

---

## Tech Stack

Python 3.11 · NumPy · SciPy · scikit-learn · PyTorch 2.x · torchao · pytest · GitHub Actions
