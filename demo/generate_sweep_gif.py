#!/usr/bin/env python3
"""
Animated GIF built entirely from the real signal_processing package:
sweeps the cutoff frequency of sp.fir_lowpass() and shows the live
filtered output using sp.fir_filter_zero_phase().
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import signal_processing as sp

demo_dir = "demo_output"
os.makedirs(demo_dir, exist_ok=True)

fs = 1000.0
duration = 1.0
t = np.arange(int(duration * fs)) / fs

np.random.seed(0)
clean = np.sin(2 * np.pi * 50 * t)
noisy = clean + 0.5 * np.random.randn(len(t))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5))
line1, = ax1.plot(t[:300], noisy[:300], color="tab:blue", linewidth=1)
line2, = ax2.plot([], [], color="tab:red", linewidth=1.6)

ax1.set_title("Input Signal (noisy, 50 Hz + noise)", fontweight="bold")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, 0.3)
ax1.set_ylim(-2.5, 2.5)
ax1.grid(True, alpha=0.3)

ax2.set_title("sp.fir_lowpass + sp.fir_filter_zero_phase output", fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.set_xlim(0, 0.3)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)

cutoff_text = ax2.text(0.02, 1.2, "", fontsize=11, color="tab:red", fontweight="bold")
plt.tight_layout()

cutoffs = np.linspace(20, 250, 24)


def animate(i):
    c = cutoffs[i]
    b, a = sp.fir_lowpass(cutoff=c, numtaps=101, fs=fs)
    out = sp.fir_filter_zero_phase(noisy, b, a)
    line2.set_data(t[:300], out[:300])
    cutoff_text.set_text(f"cutoff = {c:.0f} Hz")
    return line2, cutoff_text


anim = FuncAnimation(fig, animate, frames=len(cutoffs), interval=150, blit=True)
anim.save(f"{demo_dir}/filter_sweep_demo.gif", writer=PillowWriter(fps=6))
plt.close()

print(f"Saved {demo_dir}/filter_sweep_demo.gif")
