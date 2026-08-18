import pytest
import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import():
    """Test that we can import the package"""
    import signal_processing
    assert signal_processing.__version__ == "0.1.0"

def test_fir_lowpass():
    """Test FIR lowpass filter design"""
    from signal_processing.filters.fir import fir_lowpass
    b, a = fir_lowpass(cutoff=0.2, numtaps=11, fs=2.0)
    assert len(b) == 11
    assert np.allclose(a, [1.0])

def test_fft():
    """Test FFT function"""
    from signal_processing.transforms.fft import fft, ifft
    signal = np.array([1, 0, -1, 0])
    spectrum = fft(signal)
    recovered = ifft(spectrum)
    assert np.allclose(signal, recovered.real, atol=1e-10)

def test_am_modulate():
    """Test AM modulation"""
    from signal_processing.modulation.digital import am_modulate, am_demodulate
    fs = 1000.0
    t = np.arange(0, 1, 1/fs)
    message = np.sin(2*np.pi*5*t)  # 5 Hz tone
    modulated = am_modulate(message, carrier_freq=50.0, modulation_index=0.5, fs=fs)
    demodulated = am_demodulate(modulated, carrier_freq=50.0, fs=fs)
    # Check that we recovered something resembling the original
    correlation = np.correlate(message[:500], demodulated[:500], mode='valid')
    assert np.max(correlation) > 100  # Should have significant correlation

if __name__ == "__main__":
    test_import()
    test_fir_lowpass()
    test_fft()
    test_am_modulate()
    print("All tests passed!")
