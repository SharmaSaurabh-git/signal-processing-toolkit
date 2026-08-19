"""
Minimal example: design a low-pass FIR filter, apply it to a noisy
sine wave, and plot the before/after comparison.

Run with:
    python -m signal_processing.examples.basic_filtering
"""
import numpy as np

from ..filters import fir_lowpass, fir_filter
from ..utils import plot_signal, generate_test_signal


def main():
    fs = 500.0  # Hz
    duration = 1.0  # seconds

    print("Signal Processing Toolkit - Basic Filtering Example")
    print("Generating a 20 Hz sine wave with added noise...")

    clean = generate_test_signal("sine", duration=duration, fs=fs, frequency=20.0)
    noisy = clean + 0.4 * np.random.randn(len(clean))

    print("Designing a low-pass filter (cutoff = 40 Hz)...")
    b, a = fir_lowpass(cutoff=40.0, numtaps=51, fs=fs)

    print("Applying filter...")
    filtered = fir_filter(noisy, b, a)

    plot_signal(noisy, fs=fs, title="Noisy 20 Hz Sine Wave")
    plot_signal(filtered, fs=fs, title="Filtered Signal (Low-pass, 40 Hz cutoff)")

    print("Done.")


if __name__ == "__main__":
    main()
