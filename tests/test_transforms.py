"""Tests for signal_processing.transforms (FFT, PSD, spectrogram, STFT)."""
import numpy as np
import pytest

from signal_processing.transforms import (
    fft, ifft, fftfreq, fftshift, ifftshift,
    power_spectral_density, welch, spectrogram, stft, istft,
    coherence, correlate,
)


def test_fft_ifft_roundtrip():
    data = np.array([1, 0, -1, 0], dtype=float)
    spectrum = fft(data)
    recovered = ifft(spectrum)
    assert np.allclose(data, recovered.real, atol=1e-10)


def test_fftfreq_length():
    freqs = fftfreq(16, d=0.1)
    assert len(freqs) == 16


def test_fftshift_ifftshift_roundtrip():
    data = np.arange(8)
    assert np.array_equal(ifftshift(fftshift(data)), data)


def test_power_spectral_density_runs():
    """Regression test: crashed with AttributeError before the fix
    (parameter `signal`/`data` shadowed scipy.signal)."""
    data = np.random.randn(1000)
    freqs, psd = power_spectral_density(data, fs=1000.0, nperseg=256)
    assert len(freqs) == len(psd)
    assert np.all(psd >= 0)


def test_welch_is_alias_for_power_spectral_density():
    data = np.random.randn(500)
    f1, p1 = welch(data, fs=1000.0, nperseg=128)
    f2, p2 = power_spectral_density(data, fs=1000.0, nperseg=128)
    assert np.array_equal(f1, f2)
    assert np.array_equal(p1, p2)


def test_spectrogram_runs():
    """Regression test: crashed with AttributeError before the fix."""
    data = np.random.randn(2000)
    freqs, times, Sxx = spectrogram(data, fs=1000.0, nperseg=256)
    assert Sxx.shape == (len(freqs), len(times))


def test_stft_istft_roundtrip():
    """Regression test: stft() crashed with AttributeError before the fix."""
    fs = 1000.0
    t = np.arange(1000) / fs
    data = np.sin(2 * np.pi * 50 * t)
    freqs, times, Zxx = stft(data, fs=fs, nperseg=128)
    _, recovered = istft(Zxx, fs=fs, nperseg=128)
    n = min(len(data), len(recovered))
    assert np.allclose(data[:n], recovered[:n], atol=0.1)


def test_coherence_range():
    data = np.random.randn(1000)
    freqs, Cxy = coherence(data, data, fs=1000.0, nperseg=128)
    # A signal is perfectly coherent with itself
    assert np.allclose(Cxy, 1.0, atol=1e-6)


def test_correlate_length():
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0.5])
    result = correlate(a, b, mode="full")
    assert len(result) == len(a) + len(b) - 1
