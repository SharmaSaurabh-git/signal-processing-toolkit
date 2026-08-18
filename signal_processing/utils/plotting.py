"""
Utility functions for signal processing
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List
import scipy.signal as signal

def plot_signal(signal: np.ndarray, fs: float = 1.0,
                title: str = "Signal", 
                xlabel: str = "Time (s)",
                ylabel: str = "Amplitude",
                figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot a signal in time domain.
    """
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=figsize)
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrum(signal: np.ndarray, fs: float = 1.0,
                  title: str = "Frequency Spectrum",
                  xlabel: str = "Frequency (Hz)",
                  ylabel: str = "Magnitude",
                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot magnitude spectrum of a signal.
    """
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    magnitude = np.abs(spectrum)
    
    plt.figure(figsize=figsize)
    plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(signal: np.ndarray, fs: float = 1.0,
                     title: str = "Spectrogram",
                     xlabel: str = "Time (s)",
                     ylabel: str = "Frequency (Hz)",
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot spectrogram of a signal.
    """
    freqs, times, Sxx = signal.spectrogram(signal, fs=fs)
    plt.figure(figsize=figsize)
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label='Power (dB)')
    plt.tight_layout()
    plt.show()

def generate_test_signal(signal_type: str = 'chirp',
                         duration: float = 1.0,
                         fs: float = 1000.0,
                         **kwargs) -> np.ndarray:
    """
    Generate various test signals for experimentation.
    
    Parameters
    ----------
    signal_type : str
        Type of signal: 'sine', 'chirp', 'square', 'sawtooth', 'pulse', 'noise'
    duration : float
        Duration in seconds
    fs : float
        Sampling frequency in Hz
    **kwargs : dict
        Additional parameters specific to signal type
        
    Returns
    -------
    signal : ndarray
        Generated test signal
    """
    t = np.arange(int(duration * fs)) / fs
    
    if signal_type == 'sine':
        freq = kwargs.get('frequency', 5.0)
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == 'chirp':
        f0 = kwargs.get('f0', 0.0)
        f1 = kwargs.get('f1', fs/2)
        return signal.chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    elif signal_type == 'square':
        freq = kwargs.get('frequency', 5.0)
        return signal.square(2 * np.pi * freq * t)
    elif signal_type == 'sawtooth':
        freq = kwargs.get('frequency', 5.0)
        return signal.sawtooth(2 * np.pi * freq * t)
    elif signal_type == 'pulse':
        width = kwargs.get('width', 0.1)
        return signal.unit_impulse(len(t), idx=int(len(t)*width//2))
    elif signal_type == 'noise':
        return np.random.randn(len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

def calculate_snr(signal: np.ndarray, noise: np.ndarray = None) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    Parameters
    ----------
    signal : ndarray
        Original signal
    noise : ndarray, optional
        Noise component. If None, assumes signal contains signal+noise
        
    Returns
    -------
    snr_db : float
        SNR in decibels
    """
    if noise is None:
        # Estimate noise as high-frequency component
        # Simple approach: use wavelet or high-pass filter
        from .filters import fir_highpass
        b, a = fir_highpass(cutoff=0.1, numtaps=101, fs=1.0)
        noise = signal.lfilter(b, a, signal)
        signal_est = signal - noise
    else:
        signal_est = signal
    
    signal_power = np.mean(signal_est ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = signal_power / noise_power
    return 10 * np.log10(snr)

def eye_diagram(signal: np.ndarray, fs: float = 1.0,
                symbol_rate: float = 100.0,
                spans: int = 2) -> None:
    """
    Plot eye diagram for digital signal analysis.
    """
    samples_per_symbol = fs / symbol_rate
    if samples_per_symbol < 2:
        raise ValueError("Sampling frequency must be at least 2x symbol rate")
    
    n_samples = int(spans * samples_per_symbol)
    eye_matrix = np.zeros((n_samples, int(len(signal) / samples_per_symbol)))
    
    for i in range(eye_matrix.shape[1]):
        start_idx = int(i * samples_per_symbol)
        end_idx = start_idx + n_samples
        if end_idx <= len(signal):
            eye_matrix[:, i] = signal[start_idx:end_idx]
        else:
            # Pad with zeros if we run out of signal
            available = len(signal) - start_idx
            if available > 0:
                eye_matrix[:available, i] = signal[start_idx:]
    
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(n_samples) / fs
    plt.plot(time_axis, eye_matrix.T, 'b', alpha=0.5)
    plt.title('Eye Diagram')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
