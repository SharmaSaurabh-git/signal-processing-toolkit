"""
FIR (Finite Impulse Response) filter design and implementation
"""
import numpy as np
from typing import Tuple, Union, List
import scipy.signal as signal

def fir_lowpass(cutoff: float, numtaps: int, window: str = 'hamming',
                fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a low-pass FIR filter using the window method.
    
    Parameters
    ----------
    cutoff : float
        Cutoff frequency (normalized 0 to 1, where 1 is Nyquist frequency)
    numtaps : int
        Number of filter taps (must be odd)
    window : str
        Window function to use ('hamming', 'hann', 'blackman', etc.)
    fs : float
        Sampling frequency (default 2.0 for normalized frequency)
        
    Returns
    -------
    b : ndarray
        Numerator coefficients of the filter
    a : ndarray
        Denominator coefficients of the filter (always [1] for FIR)
    """
    if numtaps % 2 == 0:
        numtaps += 1  # Ensure odd number of taps for linear phase
        print(f"Warning: numtaps adjusted to {numtaps} (must be odd)")
        
    # Normalize cutoff frequency
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    # Design filter using scipy
    b = signal.firwin(numtaps, normalized_cutoff, window=window, fs=fs)
    a = np.array([1.0])  # FIR filter has no feedback
    
    return b, a

def fir_highpass(cutoff: float, numtaps: int, window: str = 'hamming',
                 fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a high-pass FIR filter.
    """
    if numtaps % 2 == 0:
        numtaps += 1
        
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    b = signal.firwin(numtaps, normalized_cutoff, pass_zero=False, 
                      window=window, fs=fs)
    a = np.array([1.0])
    
    return b, a

def fir_bandpass(lowcut: float, highcut: float, numtaps: int,
                 window: str = 'hamming', fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a band-pass FIR filter.
    """
    if numtaps % 2 == 0:
        numtaps += 1
        
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b = signal.firwin(numtaps, [low, high], pass_zero=False,
                      window=window, fs=fs)
    a = np.array([1.0])
    
    return b, a

def fir_bandstop(lowcut: float, highcut: float, numtaps: int,
                 window: str = 'hamming', fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a band-stop FIR filter.
    """
    if numtaps % 2 == 0:
        numtaps += 1
        
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b = signal.firwin(numtaps, [low, high], pass_zero=True,
                      window=window, fs=fs)
    a = np.array([1.0])
    
    return b, a

def fir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """
    Apply FIR filter to a signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    b : ndarray
        Filter numerator coefficients
    a : ndarray, optional
        Filter denominator coefficients (default [1] for FIR)
        
    Returns
    -------
    filtered_signal : ndarray
        Filtered output signal
    """
    if a is None:
        a = np.array([1.0])
        
    return signal.lfilter(b, a, signal)

def kaiserord(ripple: float, width: float) -> Tuple[int, float]:
    """
    Estimate filter order and beta parameter for Kaiser window.
    
    Parameters
    ----------
    ripple : float
        Desired ripple in passband (dB, positive value)
    width : float
        Width of transition region (normalized frequency)
        
    Returns
    -------
    numtaps : int
        Estimated filter order
    beta : float
        Kaiser window beta parameter
    """
    # Simple approximation - in practice would use scipy.signal.kaiserord
    # For demo purposes:
    beta = 0.1102 * (ripple - 8.7)
    numtaps = int(np.ceil((ripple - 7.95) / (2.285 * width))) + 1
    return max(numtaps, 1), max(beta, 0)
