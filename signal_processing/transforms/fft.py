"""
Signal transforms: FFT, DFT, STFT, wavelet transforms
"""
import numpy as np
from typing import Union, Tuple, Optional
import scipy.signal as signal
import scipy.fftpack as fftpack

def fft(signal: np.ndarray, n: int = None) -> np.ndarray:
    """
    Compute Fast Fourier Transform.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    n : int, optional
        Length of the transformed axis (default: length of signal)
        
    Returns
    -------
    spectrum : ndarray
        Complex FFT coefficients
    """
    return np.fft.fft(signal, n=n)

def ifft(spectrum: np.ndarray, n: int = None) -> np.ndarray:
    """
    Compute Inverse Fast Fourier Transform.
    """
    return np.fft.ifft(spectrum, n=n)

def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.
    """
    return np.fft.fftfreq(n, d=d)

def fftshift(spectrum: np.ndarray) -> np.ndarray:
    """
    Shift the zero-frequency component to the center of the spectrum.
    """
    return np.fft.fftshift(spectrum)

def ifftshift(spectrum: np.ndarray) -> np.ndarray:
    """
    Inverse of fftshift.
    """
    return np.fft.ifftshift(spectrum)

def power_spectral_density(signal: np.ndarray, fs: float = 1.0,
                          nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    fs : float
        Sampling frequency
    nperseg : int, optional
        Length of each segment
        
    Returns
    -------
    freqs : ndarray
        Array of sample frequencies
    psd : ndarray
        Power spectral density
    """
    freqs, psd = signal.welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def spectrogram(signal: np.ndarray, fs: float = 1.0,
                nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram using short-time Fourier transform.
    
    Returns
    -------
    freqs : ndarray
        Array of sample frequencies
    times : ndarray
        Array of segment times
    Sxx : ndarray
        Spectrogram of the signal
    """
    freqs, times, Sxx = signal.spectrogram(signal, fs=fs, 
                                           nperseg=nperseg, noverlap=noverlap)
    return freqs, times, Sxx

def stft(signal: np.ndarray, fs: float = 1.0,
         nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.
    """
    freqs, times, Zxx = signal.stft(signal, fs=fs, 
                                    nperseg=nperseg, noverlap=noverlap)
    return freqs, times, Zxx

def istft(Zxx: np.ndarray, fs: float = 1.0,
          nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Inverse Short-Time Fourier Transform.
    """
    t, x = signal.istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return t, x

def coherence(signal1: np.ndarray, signal2: np.ndarray,
              fs: float = 1.0, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude squared coherence between two signals.
    """
    freqs, Cxy = signal.coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    return freqs, Cxy

def correlate(signal1: np.ndarray, signal2: np.ndarray,
              mode: str = 'full') -> np.ndarray:
    """
    Cross-correlate two signals.
    """
    return signal.correlate(signal1, signal2, mode=mode)
