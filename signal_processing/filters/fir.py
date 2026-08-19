"""
FIR (Finite Impulse Response) filter design and implementation.
"""
import numpy as np
from typing import Tuple
import scipy.signal as sps


def _ensure_odd_taps(numtaps: int) -> int:
    """Force an odd tap count so the filter has exact linear phase."""
    if numtaps < 1:
        raise ValueError(f"numtaps must be a positive integer, got {numtaps}")
    if numtaps % 2 == 0:
        numtaps += 1
    return numtaps


def fir_lowpass(cutoff: float, numtaps: int, window: str = "hamming",
                 fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a low-pass FIR filter using the window method.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in the same units as `fs`.
    numtaps : int
        Number of filter taps. Forced to the next odd number if even,
        so the filter has exact linear phase.
    window : str
        Window function to use ('hamming', 'hann', 'blackman', etc.).
    fs : float
        Sampling frequency.

    Returns
    -------
    b : ndarray
        Numerator (FIR) coefficients.
    a : ndarray
        Denominator coefficients, always ``[1.0]`` for an FIR filter.
    """
    numtaps = _ensure_odd_taps(numtaps)
    if not (0 < cutoff < fs / 2.0):
        raise ValueError(f"cutoff must be between 0 and Nyquist ({fs / 2.0}), got {cutoff}")

    b = sps.firwin(numtaps, cutoff, window=window, fs=fs)
    a = np.array([1.0])
    return b, a


def fir_highpass(cutoff: float, numtaps: int, window: str = "hamming",
                  fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a high-pass FIR filter using the window method.

    Parameters are the same as :func:`fir_lowpass`.
    """
    numtaps = _ensure_odd_taps(numtaps)
    if not (0 < cutoff < fs / 2.0):
        raise ValueError(f"cutoff must be between 0 and Nyquist ({fs / 2.0}), got {cutoff}")

    b = sps.firwin(numtaps, cutoff, pass_zero=False, window=window, fs=fs)
    a = np.array([1.0])
    return b, a


def fir_bandpass(lowcut: float, highcut: float, numtaps: int,
                  window: str = "hamming", fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a band-pass FIR filter using the window method.
    """
    numtaps = _ensure_odd_taps(numtaps)
    if not (0 < lowcut < highcut < fs / 2.0):
        raise ValueError(f"require 0 < lowcut < highcut < Nyquist ({fs / 2.0}); "
                          f"got lowcut={lowcut}, highcut={highcut}")

    b = sps.firwin(numtaps, [lowcut, highcut], pass_zero=False, window=window, fs=fs)
    a = np.array([1.0])
    return b, a


def fir_bandstop(lowcut: float, highcut: float, numtaps: int,
                  window: str = "hamming", fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a band-stop (notch) FIR filter using the window method.
    """
    numtaps = _ensure_odd_taps(numtaps)
    if not (0 < lowcut < highcut < fs / 2.0):
        raise ValueError(f"require 0 < lowcut < highcut < Nyquist ({fs / 2.0}); "
                          f"got lowcut={lowcut}, highcut={highcut}")

    b = sps.firwin(numtaps, [lowcut, highcut], pass_zero=True, window=window, fs=fs)
    a = np.array([1.0])
    return b, a


def fir_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """
    Apply an FIR (or general IIR) filter to a signal.

    Note on group delay: a linear-phase FIR filter with N taps delays
    its output by exactly (N - 1) / 2 samples relative to the input.
    If you overlay `data` and the returned signal to compare shapes,
    they will appear shifted — that is expected, not a bug. Use
    `fir_filter_zero_phase` instead if you need the output aligned
    in time with the input (e.g. for visual comparison).

    Parameters
    ----------
    data : ndarray
        Input signal.
    b : ndarray
        Filter numerator coefficients.
    a : ndarray, optional
        Filter denominator coefficients. Defaults to ``[1.0]`` for FIR.

    Returns
    -------
    ndarray
        Filtered output signal, same length as `data`, delayed by the
        filter's group delay.
    """
    if a is None:
        a = np.array([1.0])
    return sps.lfilter(b, a, data)


def fir_filter_zero_phase(data: np.ndarray, b: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """
    Apply an FIR (or general IIR) filter with zero phase distortion.

    Filters the signal forward and backward (`scipy.signal.filtfilt`),
    which cancels out group delay so the output stays time-aligned
    with the input. This is usually what you want for offline
    analysis and plotting; use `fir_filter` instead for streaming/
    real-time applications where only causal filtering is possible.

    Parameters
    ----------
    data : ndarray
        Input signal.
    b : ndarray
        Filter numerator coefficients.
    a : ndarray, optional
        Filter denominator coefficients. Defaults to ``[1.0]`` for FIR.

    Returns
    -------
    ndarray
        Zero-phase filtered signal, same length as `data`, time-aligned
        with the input.
    """
    if a is None:
        a = np.array([1.0])
    return sps.filtfilt(b, a, data)


def kaiserord(ripple: float, width: float) -> Tuple[int, float]:
    """
    Estimate FIR filter order and Kaiser window beta for a given
    passband ripple / stopband attenuation and transition width.

    Thin wrapper around ``scipy.signal.kaiserord`` so the toolkit's
    public API stays self-contained.

    Parameters
    ----------
    ripple : float
        Desired attenuation in the stop band, in dB (positive value).
    width : float
        Width of the transition region, normalized so 1.0 corresponds
        to the Nyquist frequency.

    Returns
    -------
    numtaps : int
        Estimated filter length (order + 1).
    beta : float
        Kaiser window shape parameter.
    """
    numtaps, beta = sps.kaiserord(ripple, width)
    return numtaps, beta
