"""
Modulation and demodulation techniques for communication systems
"""
import numpy as np
from typing import Tuple, Union, Optional
import scipy.signal as signal
import scipy.fftpack as fftpack

def am_modulate(message: np.ndarray, carrier_freq: float, 
                modulation_index: float = 0.5, 
                fs: float = 1000.0) -> np.ndarray:
    """
    Amplitude Modulation (AM).
    
    Parameters
    ----------
    message : ndarray
        Message signal (normalized to [-1, 1])
    carrier_freq : float
        Carrier frequency in Hz
    modulation_index : float
        Modulation index (0 to 1)
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    modulated : ndarray
        AM modulated signal
    """
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    modulated = (1 + modulation_index * message) * carrier
    return modulated

def am_demodulate(am_signal, carrier_freq, fs=1000.0):
    """
    AM demodulation using envelope detection.
    """

    analytic_signal = signal.hilbert(am_signal)

    envelope = np.abs(analytic_signal)

    envelope = envelope - np.mean(envelope)

    return envelope

def fm_modulate(message: np.ndarray, carrier_freq: float,
                frequency_deviation: float, 
                fs: float = 1000.0) -> np.ndarray:
    """
    Frequency Modulation (FM).
    
    Parameters
    ----------
    message : ndarray
        Message signal
    carrier_freq : float
        Carrier frequency in Hz
    frequency_deviation : float
        Peak frequency deviation in Hz
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    modulated : ndarray
        FM modulated signal
    """
    t = np.arange(len(message)) / fs
    # Integrate message to get phase deviation
    phase_deviation = 2 * np.pi * frequency_deviation * np.cumsum(message) / fs
    instantaneous_phase = 2 * np.pi * carrier_freq * t + phase_deviation
    modulated = np.cos(instantaneous_phase)
    return modulated

def fm_demodulate(fm_signal: np.ndarray, carrier_freq: float,
                  frequency_deviation: float,
                  fs: float = 1000.0) -> np.ndarray:
    """
    FM demodulation using phase-locked loop (simplified).
    """
    # Simple approach: differentiate phase
    analytic_signal = signal.hilbert(fm_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2 * np.pi)
    
    # Center around carrier frequency
    message = instantaneous_frequency - carrier_freq
    
    # Scale by frequency deviation
    message = message / frequency_deviation
    
    return message

def psk_modulate(bits: np.ndarray, carrier_freq: float,
                 fs: float = 1000.0, 
                 samples_per_symbol: int = 16) -> np.ndarray:
    """
    Binary Phase Shift Keying (BPSK) modulation.
    """
    # Map bits to symbols: 0 -> -1, 1 -> +1
    symbols = 2 * bits.astype(float) - 1
    
    # Upsample
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    # Generate carrier
    t = np.arange(len(upsampled)) / fs
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    # Modulate
    modulated = upsampled * carrier
    return modulated

def qpsk_modulate(bits: np.ndarray, carrier_freq: float,
                  fs: float = 1000.0,
                  samples_per_symbol: int = 16) -> np.ndarray:
    """
    Quadrature Phase Shift Keying (QPSK) modulation.
    """
    # Ensure even number of bits
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # Pad with zero if odd
    
    # Group bits into pairs and map to symbols
    symbol_map = {
        (0, 0): 1+1j,    # 45 degrees
        (0, 1): -1+1j,   # 135 degrees
        (1, 0): 1-1j,    # -45 degrees
        (1, 1): -1-1j    # -135 degrees
    }
    
    symbols = []
    for i in range(0, len(bits), 2):
        bit_pair = (bits[i], bits[i+1])
        symbols.append(symbol_map[bit_pair])
    
    symbols = np.array(symbols)
    
    # Upsample
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    # Generate carrier (complex)
    t = np.arange(len(upsampled)) / fs
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    # Modulate
    modulated = upsampled * carrier
    
    # Return real part for transmission
    return np.real(modulated)

def generate_pn_sequence(length: int, taps: list) -> np.ndarray:
    """
    Generate Pseudo-Noise (PN) sequence using LFSR.
    
    Parameters
    ----------
    length : int
        Length of PN sequence to generate
    taps : list
        Tap positions for LFSR (e.g., [5, 2] for x^5 + x^2 + 1)
        
    Returns
    -------
    pn_sequence : ndarray
        Binary PN sequence (±1)
    """
    # Initialize LFSR with all ones
    register = np.ones(max(taps), dtype=int)
    sequence = []
    
    for _ in range(length):
        # Output is the last bit
        output = register[-1]
        sequence.append(1 if output == 1 else -1)  # Map to ±1
        
        # Compute feedback
        feedback = 0
        for tap in taps:
            feedback ^= register[tap - 1]  # Adjust for 0-indexing
        
        # Shift register
        register = np.roll(register, 1)
        register[0] = feedback
    
    return np.array(sequence)
