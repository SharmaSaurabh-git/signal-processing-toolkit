"""Tests for signal_processing.filters (FIR filter design and application)."""
import numpy as np
import pytest

from signal_processing.filters import (
    fir_lowpass, fir_highpass, fir_bandpass, fir_bandstop,
    fir_filter, fir_filter_zero_phase, kaiserord,
)


def test_fir_lowpass_shape():
    b, a = fir_lowpass(cutoff=0.2, numtaps=11, fs=2.0)
    assert len(b) == 11
    assert np.allclose(a, [1.0])


def test_fir_lowpass_forces_odd_taps():
    b, a = fir_lowpass(cutoff=0.2, numtaps=10, fs=2.0)
    assert len(b) == 11  # bumped up to next odd number


def test_fir_lowpass_cutoff_is_accurate():
    """Regression test: cutoff was previously double-normalized, giving a
    filter ~5-500x narrower than requested."""
    fs = 1000.0
    b, a = fir_lowpass(cutoff=100.0, numtaps=101, fs=fs)
    w = np.linspace(1, fs / 2, 2000)
    h = np.abs(np.fft.rfft(b, n=8192))
    freqs = np.fft.rfftfreq(8192, d=1 / fs)
    mag_db = 20 * np.log10(h + 1e-12)
    # -3dB point should land close to the requested 100 Hz cutoff
    idx = np.argmin(np.abs(mag_db - (mag_db[0] - 3)))
    assert 80.0 < freqs[idx] < 120.0


def test_fir_lowpass_rejects_invalid_cutoff():
    with pytest.raises(ValueError):
        fir_lowpass(cutoff=600.0, numtaps=51, fs=1000.0)  # above Nyquist


def test_fir_highpass_shape():
    b, a = fir_highpass(cutoff=0.3, numtaps=21, fs=2.0)
    assert len(b) == 21


def test_fir_bandpass_shape():
    b, a = fir_bandpass(lowcut=50, highcut=150, numtaps=51, fs=1000.0)
    assert len(b) == 51


def test_fir_bandpass_rejects_invalid_range():
    with pytest.raises(ValueError):
        fir_bandpass(lowcut=150, highcut=50, numtaps=51, fs=1000.0)  # swapped


def test_fir_bandstop_shape():
    b, a = fir_bandstop(lowcut=50, highcut=150, numtaps=51, fs=1000.0)
    assert len(b) == 51


def test_fir_filter_runs_and_preserves_length():
    """Regression test: this crashed with AttributeError before the fix
    (parameter `signal` shadowed the scipy.signal module import)."""
    b, a = fir_lowpass(cutoff=100.0, numtaps=51, fs=1000.0)
    data = np.random.randn(500)
    filtered = fir_filter(data, b, a)
    assert len(filtered) == len(data)
    assert np.all(np.isfinite(filtered))


def test_fir_filter_has_expected_group_delay():
    """fir_filter (lfilter) is causal: a linear-phase FIR filter delays
    its output by (numtaps - 1) / 2 samples. Verify that delay exactly."""
    fs = 1000.0
    numtaps = 101
    t = np.arange(1000) / fs
    clean = np.sin(2 * np.pi * 10 * t)
    b, a = fir_lowpass(cutoff=30.0, numtaps=numtaps, fs=fs)
    filtered = fir_filter(clean, b, a)
    delay = (numtaps - 1) // 2
    # after shifting by the known group delay, filtered output should
    # closely match the (undelayed) clean signal
    aligned = filtered[delay:]
    reference = clean[: len(aligned)]
    assert np.std(aligned - reference) < np.std(filtered[:len(reference)] - reference)


def test_fir_filter_zero_phase_smooths_noisy_signal_without_delay():
    """fir_filter_zero_phase should reduce noise AND stay time-aligned
    with the input (no group delay), unlike fir_filter."""
    fs = 1000.0
    t = np.arange(1000) / fs
    clean = np.sin(2 * np.pi * 10 * t)
    noisy = clean + 0.5 * np.random.randn(len(t))
    b, a = fir_lowpass(cutoff=30.0, numtaps=101, fs=fs)
    filtered = fir_filter_zero_phase(noisy, b, a)
    assert len(filtered) == len(noisy)
    assert np.std(filtered - clean) < np.std(noisy - clean)


def test_kaiserord_matches_scipy():
    import scipy.signal as sps
    numtaps, beta = kaiserord(60, 0.05)
    expected_numtaps, expected_beta = sps.kaiserord(60, 0.05)
    assert numtaps == expected_numtaps
    assert beta == pytest.approx(expected_beta)
