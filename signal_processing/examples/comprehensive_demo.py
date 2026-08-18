"""
Comprehensive example demonstrating the Signal Processing Toolkit
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import our toolkit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import signal_processing as sp

def main():
    print("Signal Processing Toolkit Demo")
    print("=" * 40)
    
    # Generate test signals
    fs = 1000.0  # Sampling frequency
    duration = 1.0  # seconds
    t = np.arange(int(duration * fs)) / fs
    
    # 1. Generate a test signal (chirp + noise)
    print("Generating test signal...")
    clean_signal = sp.generate_test_signal('chirp', duration=duration, fs=fs, f0=50, f1=250)
    noise = 0.5 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # 2. Plot original and noisy signals
    sp.plot_signal(noisy_signal, fs=fs, title="Noisy Chirp Signal", 
                   ylabel="Amplitude")
    
    # 3. Design and apply a low-pass filter to remove noise
    print("Designing low-pass filter...")
    cutoff_freq = 100.0  # Hz
    numtaps = 101
    b, a = sp.fir_lowpass(cutoff=cutoff_freq, numtaps=numtaps, fs=fs)
    
    filtered_signal = sp.fir_filter(noisy_signal, b, a)
    
    # 4. Plot filtered signal
    sp.plot_signal(filtered_signal, fs=fs, title="Filtered Signal (Low-pass)",
                   ylabel="Amplitude")
    
    # 5. Show frequency spectrum
    sp.plot_spectrum(noisy_signal, fs=fs, title="Noisy Signal Spectrum")
    sp.plot_spectrum(filtered_signal, fs=fs, title="Filtered Signal Spectrum")
    
    # 6. Demonstrate modulation
    print("
Demonstrating AM modulation...")
    message = np.sin(2 * np.pi * 5 * t)  # 5 Hz message
    am_signal = sp.am_modulate(message, carrier_freq=50.0, modulation_index=0.8, fs=fs)
    demodulated = sp.am_demodulate(am_signal, carrier_freq=50.0, fs=fs)
    
    # Plot modulation results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t[:200], message[:200])
    plt.title('Original Message (5 Hz sine wave)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:200], am_signal[:200])
    plt.title('AM Modulated Signal (carrier=50 Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t[:200], demodulated[:200])
    plt.title('Demodulated Message')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 7. Calculate SNR improvement
    snr_before = sp.calculate_snr(clean_signal, noise)
    snr_after = sp.calculate_snr(clean_signal, noisy_signal - filtered_signal)
    print(f"
SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_after - snr_before:.2f} dB")
    
    # 8. Generate and plot a PN sequence
    print("
Generating PN sequence...")
    pn_seq = sp.generate_pn_sequence(length=63, taps=[6, 1])  # x^6 + x + 1
    sp.plot_signal(pn_seq[:20], fs=fs*10, title="PN Sequence (first 20 chips)",
                   xlabel="Chip number", ylabel="Amplitude")
    
    print("
Demo completed successfully!")

if __name__ == "__main__":
    main()
