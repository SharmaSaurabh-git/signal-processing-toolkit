#!/usr/bin/env python3
"""
Extra showcase plots built entirely from the real signal_processing package:
- QPSK constellation diagram (via real coherent demodulation using fir_lowpass)
- Eye diagram (BPSK, using the library's own eye_diagram function)
- FFT magnitude spectrum (using sp.fft / sp.fftfreq / sp.fftshift)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import signal_processing as sp

demo_dir = "demo_output"
os.makedirs(demo_dir, exist_ok=True)

# ---------------------------------------------------------------
# 1. QPSK constellation - real coherent demodulation of the ACTUAL
#    qpsk_modulate() output, using the library's own fir_lowpass +
#    fir_filter for I/Q recovery. Nothing here is simulated separately;
#    it is literally decoding the package's own modulated waveform.
# ---------------------------------------------------------------
print("1. QPSK constellation via real coherent demodulation...")
np.random.seed(7)
fs = 2000.0
carrier_freq = 100.0
sps_ = 32
n_symbols = 400
bits = np.random.randint(0, 2, n_symbols * 2)

qpsk_signal = sp.qpsk_modulate(bits, carrier_freq=carrier_freq, fs=fs,
                                samples_per_symbol=sps_)

t = np.arange(len(qpsk_signal)) / fs
i_mixed = qpsk_signal * np.cos(2 * np.pi * carrier_freq * t)
q_mixed = qpsk_signal * -np.sin(2 * np.pi * carrier_freq * t)

b_lp, a_lp = sp.fir_lowpass(cutoff=carrier_freq / 2, numtaps=101, fs=fs)
# zero-phase filtering avoids the causal filter's group delay, which was
# smearing the recovered symbol timing and scattering the constellation
i_base = sp.fir_filter_zero_phase(i_mixed, b_lp, a_lp)
q_base = sp.fir_filter_zero_phase(q_mixed, b_lp, a_lp)

# sample at symbol centers (skip first/last symbol to avoid filter edge effects)
offset = sps_ // 2
i_syms = i_base[offset::sps_][1:-1]
q_syms = q_base[offset::sps_][1:-1]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(i_syms, q_syms, s=14, alpha=0.55, color="#1f77b4", edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_title("QPSK Constellation\n(recovered via sp.fir_lowpass coherent demod)",
              fontsize=13, fontweight="bold")
ax.set_xlabel("In-phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{demo_dir}/qpsk_constellation.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 2. Eye diagram - real BPSK signal through the library's own
#    eye_diagram() plotting function.
# ---------------------------------------------------------------
print("2. Eye diagram from real BPSK signal...")
np.random.seed(3)
bpsk_bits = np.random.randint(0, 2, 200)
bpsk_sps = 20
bpsk_signal = sp.bpsk_modulate(bpsk_bits, carrier_freq=50.0, fs=1000.0,
                                samples_per_symbol=bpsk_sps)

# Get baseband envelope for a cleaner classic eye pattern via Hilbert-style
# demod already provided by the library (am_demodulate reuses hilbert envelope
# logic internally); here we directly eye-diagram the raw modulated waveform's
# lowpassed baseband using the library's own filter, matching the toolkit's
# own vocabulary (fir_lowpass + fir_filter).
b_env, a_env = sp.fir_lowpass(cutoff=40.0, numtaps=101, fs=1000.0)
baseband = sp.fir_filter(bpsk_signal * np.cos(2 * np.pi * 50.0 * np.arange(len(bpsk_signal)) / 1000.0), b_env, a_env)

sp.eye_diagram(baseband, fs=1000.0, symbol_rate=1000.0 / bpsk_sps, spans=2)
plt.savefig(f"{demo_dir}/eye_diagram.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 3. FFT magnitude spectrum - using sp.fft / sp.fftfreq / sp.fftshift
# ---------------------------------------------------------------
print("3. FFT magnitude spectrum...")
fs2 = 1000.0
t2 = np.arange(1000) / fs2
sig = np.sin(2*np.pi*50*t2) + 0.6*np.sin(2*np.pi*120*t2) + 0.3*np.random.randn(len(t2))

spectrum = sp.fftshift(sp.fft(sig))
freqs = sp.fftshift(sp.fftfreq(len(sig), d=1/fs2))
mag = np.abs(spectrum) / len(sig)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(freqs, mag, color="#d62728", linewidth=1.1)
ax.set_xlim(0, 200)
ax.set_title("FFT Magnitude Spectrum (sp.fft) - 50 Hz + 120 Hz components",
              fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{demo_dir}/fft_spectrum.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nExtra plots saved to demo_output/:")
for f in ["qpsk_constellation.png", "eye_diagram.png", "fft_spectrum.png"]:
    print(f"  - {f}")
