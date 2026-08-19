"""
Utility functions: test-signal generation, plotting, and signal metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import scipy.signal as sps


def plot_signal(data: np.ndarray, fs: float = 1.0,
                 title: str = "Signal",
                 xlabel: str = "Time (s)",
                 ylabel: str = "Amplitude",
                 figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot a signal in the time domain."""
    t = np.arange(len(data)) / fs
    plt.figure(figsize=figsize)
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spectrum(data: np.ndarray, fs: float = 1.0,
                   title: str = "Frequency Spectrum",
                   xlabel: str = "Frequency (Hz)",
                   ylabel: str = "Magnitude",
                   figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot the (single-sided) magnitude spectrum of a signal."""
    spectrum = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1 / fs)
    magnitude = np.abs(spectrum)

    plt.figure(figsize=figsize)
    plt.plot(freqs[:len(freqs) // 2], magnitude[:len(magnitude) // 2])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(data: np.ndarray, fs: float = 1.0,
                      title: str = "Spectrogram",
                      xlabel: str = "Time (s)",
                      ylabel: str = "Frequency (Hz)",
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot the spectrogram of a signal."""
    freqs, times, Sxx = sps.spectrogram(data, fs=fs)
    plt.figure(figsize=figsize)
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()


def generate_test_signal(signal_type: str = "chirp",
                          duration: float = 1.0,
                          fs: float = 1000.0,
                          **kwargs) -> np.ndarray:
    """
    Generate common test signals for experimentation.

    Parameters
    ----------
    signal_type : str
        One of 'sine', 'chirp', 'square', 'sawtooth', 'pulse', 'noise'.
    duration : float
        Duration in seconds.
    fs : float
        Sampling frequency in Hz.
    **kwargs : dict
        Extra parameters specific to `signal_type`
        (e.g. `frequency=`, `f0=`, `f1=`, `width=`).

    Returns
    -------
    ndarray
        Generated test signal.
    """
    t = np.arange(int(duration * fs)) / fs

    if signal_type == "sine":
        freq = kwargs.get("frequency", 5.0)
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == "chirp":
        f0 = kwargs.get("f0", 0.0)
        f1 = kwargs.get("f1", fs / 2)
        return sps.chirp(t, f0=f0, f1=f1, t1=duration, method="linear")
    elif signal_type == "square":
        freq = kwargs.get("frequency", 5.0)
        return sps.square(2 * np.pi * freq * t)
    elif signal_type == "sawtooth":
        freq = kwargs.get("frequency", 5.0)
        return sps.sawtooth(2 * np.pi * freq * t)
    elif signal_type == "pulse":
        width = kwargs.get("width", 0.1)
        return sps.unit_impulse(len(t), idx=int(len(t) * width // 2))
    elif signal_type == "noise":
        return np.random.randn(len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type!r}")


def calculate_snr(data: np.ndarray, noise: np.ndarray = None) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    data : ndarray
        Signal to evaluate. If `noise` is None, `data` is treated as a
        noisy signal and its noise floor is estimated with a high-pass
        filter.
    noise : ndarray, optional
        Known noise component, same length as `data`.

    Returns
    -------
    float
        SNR in decibels. Returns ``inf`` if the noise power is zero.
    """
    if noise is None:
        from ..filters.fir import fir_highpass, fir_filter
        b, a = fir_highpass(cutoff=0.1, numtaps=101, fs=1.0)
        noise = fir_filter(data, b, a)
        signal_est = data - noise
    else:
        if len(noise) != len(data):
            raise ValueError(
                f"data and noise must be the same length, "
                f"got {len(data)} and {len(noise)}"
            )
        signal_est = data

    signal_power = np.mean(signal_est ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def eye_diagram(data: np.ndarray, fs: float = 1.0,
                 symbol_rate: float = 100.0,
                 spans: int = 2) -> None:
    """Plot an eye diagram for digital signal analysis."""
    samples_per_symbol = fs / symbol_rate
    if samples_per_symbol < 2:
        raise ValueError("Sampling frequency must be at least 2x symbol rate")

    n_samples = int(spans * samples_per_symbol)
    eye_matrix = np.zeros((n_samples, int(len(data) / samples_per_symbol)))

    for i in range(eye_matrix.shape[1]):
        start_idx = int(i * samples_per_symbol)
        end_idx = start_idx + n_samples
        if end_idx <= len(data):
            eye_matrix[:, i] = data[start_idx:end_idx]
        else:
            available = len(data) - start_idx
            if available > 0:
                eye_matrix[:available, i] = data[start_idx:]

    plt.figure(figsize=(10, 6))
    time_axis = np.arange(n_samples) / fs
    # eye_matrix has shape (n_samples, n_symbols): each column is one
    # symbol-length trace. Do NOT transpose — matplotlib plots each
    # column against time_axis when the first dimension matches.
    plt.plot(time_axis, eye_matrix, "b", alpha=0.5)
    plt.title("Eye Diagram")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
