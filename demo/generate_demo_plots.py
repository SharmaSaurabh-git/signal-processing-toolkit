#!/usr/bin/env python3
"""
Demo script that runs the Signal Processing Toolkit end-to-end and
saves the resulting plots to disk. Useful for showcasing the project
in a README or portfolio without needing a GUI.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: works over SSH/CI with no display
import matplotlib.pyplot as plt

import signal_processing as sp


def main():
    print("Signal Processing Toolkit Demo - Generating showcase plots...")

    demo_dir = "demo_output"
    os.makedirs(demo_dir, exist_ok=True)

    fs = 1000.0
    duration = 1.0
    t = np.arange(int(duration * fs)) / fs

    # 1. Filter design and application
    print("1. Designing and applying low-pass filter...")
    clean = np.sin(2 * np.pi * 50 * t)
    noise = 0.3 * np.random.randn(len(t))
    noisy = clean + noise

    b, a = sp.fir_lowpass(cutoff=100.0, numtaps=101, fs=fs)
    filtered = sp.fir_filter(noisy, b, a)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    ax1.plot(t[:200], clean[:200], "g-", linewidth=2, label="Clean Signal")
    ax1.plot(t[:200], noisy[:200], "b-", alpha=0.7, label="Noisy Signal")
    ax1.set_title("Signal in Time Domain", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t[:200], filtered[:200], "r-", linewidth=2, label="Filtered Signal")
    ax2.set_title("After Low-Pass Filtering (Cutoff: 100 Hz)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    freqs, psd = sp.power_spectral_density(noisy, fs=fs, nperseg=256)
    _, filtered_psd = sp.power_spectral_density(filtered, fs=fs, nperseg=256)

    ax3.semilogy(freqs[:len(freqs) // 2], psd[:len(psd) // 2], "b-", alpha=0.7, label="Noisy")
    ax3.semilogy(freqs[:len(freqs) // 2], filtered_psd[:len(filtered_psd) // 2], "r-", linewidth=2, label="Filtered")
    ax3.set_title("Power Spectral Density", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (V\u00b2/Hz)")
    ax3.set_xlim(0, fs / 2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{demo_dir}/filter_demo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. AM modulation
    print("2. Demonstrating AM modulation...")
    message = np.sin(2 * np.pi * 5 * t)
    carrier_freq = 50.0
    modulated = sp.am_modulate(message, carrier_freq=carrier_freq, modulation_index=0.8, fs=fs)
    demodulated = sp.am_demodulate(modulated, carrier_freq=carrier_freq, fs=fs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    ax1.plot(t[:200], message[:200], "g-", linewidth=2)
    ax1.set_title("Original Message (5 Hz)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t[:400], modulated[:400], "b-", linewidth=1)
    ax2.set_title(f"AM Modulated Signal (Carrier: {carrier_freq} Hz)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t[:200], demodulated[:200], "r-", linewidth=2)
    ax3.set_title("Demodulated Message", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{demo_dir}/modulation_demo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Spectrogram
    print("3. Generating spectrogram of chirp signal...")
    chirp = sp.generate_test_signal("chirp", duration=duration, fs=fs, f0=20, f1=200)
    noisy_chirp = chirp + 0.2 * np.random.randn(len(t))

    plt.figure(figsize=(12, 6))
    sp.plot_spectrogram(noisy_chirp, fs=fs, title="Spectrogram: Noisy Linear Chirp (20Hz \u2192 200Hz)")
    plt.savefig(f"{demo_dir}/spectrogram_demo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. PN sequence
    print("4. Generating PN sequence...")
    pn_seq = sp.generate_pn_sequence(length=127, taps=[6, 1])

    plt.figure(figsize=(12, 4))
    plt.stem(range(20), pn_seq[:20], basefmt=" ")
    plt.title("PN Sequence (First 20 Chips) - x\u2076 + x + 1", fontsize=14, fontweight="bold")
    plt.xlabel("Chip Number")
    plt.ylabel("Amplitude (\u00b11)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{demo_dir}/pn_sequence_demo.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDemo plots saved to ./{demo_dir}/")
    print("Generated files:")
    for f in sorted(os.listdir(demo_dir)):
        print(f"  - {f}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
