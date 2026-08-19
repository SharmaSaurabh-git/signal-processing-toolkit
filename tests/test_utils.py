"""Tests for signal_processing.utils (test-signal generation, SNR, plotting)."""
import matplotlib
matplotlib.use("Agg")  # headless-safe for CI

import numpy as np
import pytest

from signal_processing.utils import (
    plot_signal, plot_spectrum, plot_spectrogram,
    generate_test_signal, calculate_snr, eye_diagram,
)


@pytest.mark.parametrize("signal_type", ["sine", "chirp", "square", "sawtooth", "pulse", "noise"])
def test_generate_test_signal_types(signal_type):
    data = generate_test_signal(signal_type, duration=0.5, fs=1000.0)
    assert len(data) == 500
    assert np.all(np.isfinite(data))


def test_generate_test_signal_rejects_unknown_type():
    with pytest.raises(ValueError):
        generate_test_signal("not_a_real_type")


def test_calculate_snr_with_zero_noise_returns_inf():
    signal_power_data = np.ones(1000)
    noise = np.zeros(1000)
    snr = calculate_snr(signal_power_data, noise)
    assert snr == float("inf")


def test_calculate_snr_with_nonzero_noise():
    rng = np.random.default_rng(0)
    clean = np.sin(2 * np.pi * 5 * np.arange(1000) / 1000)
    noise = 0.1 * rng.standard_normal(1000)
    snr = calculate_snr(clean, noise)
    assert snr > 0  # signal power should dominate noise power here


def test_calculate_snr_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_snr(np.zeros(100), np.zeros(50))


def test_calculate_snr_auto_estimate_runs():
    """Regression test: crashed before the fix (scipy.signal shadowing
    AND a broken relative import: `from .filters import fir_highpass`)."""
    data = np.random.randn(500)
    snr = calculate_snr(data)  # noise=None triggers auto-estimation path
    assert np.isfinite(snr) or snr == float("inf")


def test_plot_signal_runs_without_error():
    plot_signal(np.random.randn(100), fs=100.0)


def test_plot_spectrum_runs_without_error():
    plot_spectrum(np.random.randn(100), fs=100.0)


def test_plot_spectrogram_runs_without_error():
    """Regression test: crashed with AttributeError before the fix."""
    plot_spectrogram(np.random.randn(2000), fs=1000.0)


def test_eye_diagram_runs_without_error():
    fs = 1000.0
    data = generate_test_signal("square", duration=1.0, fs=fs, frequency=10.0)
    eye_diagram(data, fs=fs, symbol_rate=100.0)


def test_eye_diagram_rejects_undersampled_signal():
    with pytest.raises(ValueError):
        eye_diagram(np.random.randn(100), fs=10.0, symbol_rate=100.0)
