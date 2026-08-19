"""
Comprehensive example demonstrating the Signal Processing Toolkit:
filtering, spectral analysis, AM modulation, SNR measurement, and
PN sequence generation, all in one script.

Run with:
    python -m signal_processing.examples.comprehensive_demo
"""
import numpy as np
import matplotlib.pyplot as plt

import signal_processing as sp


def main():
    print("Signal Processing Toolkit Demo")
    print("=" * 40)

    fs = 1000.0
    duration = 1.0
    t = np.arange(int(duration * fs)) / fs

    # 1. Generate a test signal (chirp + noise)
    # Chirp sweeps 10-60 Hz, comfortably inside the 100 Hz filter
    # passband below, so filtering removes noise without destroying
    # real signal content (a wider sweep would do both, muddying the
    # SNR comparison in step 7).
    print("Generating test signal...")
    clean_signal = sp.generate_test_signal("chirp", duration=duration, fs=fs, f0=10, f1=60)
    noise = 0.5 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise

    # 2. Plot original and noisy signals
    sp.plot_signal(noisy_signal, fs=fs, title="Noisy Chirp Signal", ylabel="Amplitude")

    # 3. Design and apply a low-pass filter to remove noise
    print("Designing low-pass filter...")
    cutoff_freq = 100.0
    numtaps = 101
    b, a = sp.fir_lowpass(cutoff=cutoff_freq, numtaps=numtaps, fs=fs)
    # Zero-phase filtering keeps the output time-aligned with the input,
    # which matters here since we compare filtered_signal directly
    # against clean_signal below (see fir_filter's docstring for why
    # the causal fir_filter would introduce a group delay instead).
    filtered_signal = sp.fir_filter_zero_phase(noisy_signal, b, a)

    # 4. Plot filtered signal
    sp.plot_signal(filtered_signal, fs=fs, title="Filtered Signal (Low-pass)", ylabel="Amplitude")

    # 5. Show frequency spectrum
    sp.plot_spectrum(noisy_signal, fs=fs, title="Noisy Signal Spectrum")
    sp.plot_spectrum(filtered_signal, fs=fs, title="Filtered Signal Spectrum")

    # 6. Demonstrate AM modulation
    print("\nDemonstrating AM modulation...")
    message = np.sin(2 * np.pi * 5 * t)
    am_signal = sp.am_modulate(message, carrier_freq=50.0, modulation_index=0.8, fs=fs)
    demodulated = sp.am_demodulate(am_signal, carrier_freq=50.0, fs=fs)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t[:200], message[:200])
    plt.title("Original Message (5 Hz sine wave)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t[:200], am_signal[:200])
    plt.title("AM Modulated Signal (carrier=50 Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t[:200], demodulated[:200])
    plt.title("Demodulated Message")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 7. Calculate SNR improvement
    snr_before = sp.calculate_snr(clean_signal, noise)
    snr_after = sp.calculate_snr(clean_signal, noisy_signal - filtered_signal)
    print(f"\nSNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_after - snr_before:.2f} dB")

    # 8. Generate and plot a PN sequence
    print("\nGenerating PN sequence...")
    pn_seq = sp.generate_pn_sequence(length=63, taps=[6, 1])
    sp.plot_signal(pn_seq[:20], fs=fs * 10, title="PN Sequence (first 20 chips)",
                    xlabel="Chip number", ylabel="Amplitude")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
