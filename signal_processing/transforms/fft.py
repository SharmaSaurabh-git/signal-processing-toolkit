"""
Signal transforms: FFT, spectral density, spectrogram, STFT, and
related frequency-domain analysis tools.
"""
import numpy as np
from typing import Tuple
import scipy.signal as sps


def fft(data: np.ndarray, n: int = None) -> np.ndarray:
    """
    Compute the Fast Fourier Transform of a signal.

    Parameters
    ----------
    data : ndarray
        Input signal.
    n : int, optional
        Length of the transformed axis. Defaults to ``len(data)``.

    Returns
    -------
    ndarray
        Complex FFT coefficients.
    """
    return np.fft.fft(data, n=n)


def ifft(spectrum: np.ndarray, n: int = None) -> np.ndarray:
    """Compute the Inverse Fast Fourier Transform."""
    return np.fft.ifft(spectrum, n=n)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Return the DFT sample frequencies for a length-`n` signal."""
    return np.fft.fftfreq(n, d=d)


def fftshift(spectrum: np.ndarray) -> np.ndarray:
    """Shift the zero-frequency component to the center of the spectrum."""
    return np.fft.fftshift(spectrum)


def ifftshift(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of :func:`fftshift`."""
    return np.fft.ifftshift(spectrum)


def power_spectral_density(data: np.ndarray, fs: float = 1.0,
                            nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    data : ndarray
        Input signal.
    fs : float
        Sampling frequency.
    nperseg : int, optional
        Length of each averaging segment.

    Returns
    -------
    freqs : ndarray
        Sample frequencies.
    psd : ndarray
        Power spectral density estimate.
    """
    freqs, psd = sps.welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd


# Backward/documentation-compatible alias — doc/api.md and common DSP
# convention both call this "welch".
welch = power_spectral_density


def spectrogram(data: np.ndarray, fs: float = 1.0,
                 nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a spectrogram via short-time Fourier transform magnitude.

    Returns
    -------
    freqs : ndarray
    times : ndarray
    Sxx : ndarray
        Spectrogram (power) of the signal.
    """
    freqs, times, Sxx = sps.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, Sxx


def stft(data: np.ndarray, fs: float = 1.0,
         nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Short-Time Fourier Transform."""
    freqs, times, Zxx = sps.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, Zxx


def istft(Zxx: np.ndarray, fs: float = 1.0,
          nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Inverse Short-Time Fourier Transform."""
    t, x = sps.istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return t, x


def coherence(signal1: np.ndarray, signal2: np.ndarray,
              fs: float = 1.0, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the magnitude-squared coherence between two signals."""
    freqs, Cxy = sps.coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    return freqs, Cxy


def correlate(signal1: np.ndarray, signal2: np.ndarray, mode: str = "full") -> np.ndarray:
    """Cross-correlate two signals."""
    return sps.correlate(signal1, signal2, mode=mode)
