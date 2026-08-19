# API Documentation

## signal_processing.filters

### FIR Filter Design

- `fir_lowpass(cutoff, numtaps, window='hamming', fs=2.0)` → `(b, a)`
- `fir_highpass(cutoff, numtaps, window='hamming', fs=2.0)` → `(b, a)`
- `fir_bandpass(lowcut, highcut, numtaps, window='hamming', fs=2.0)` → `(b, a)`
- `fir_bandstop(lowcut, highcut, numtaps, window='hamming', fs=2.0)` → `(b, a)`
- `fir_filter(data, b, a=None)` → `filtered_signal`
  Causal filtering (`scipy.signal.lfilter`). Introduces a group delay of
  `(numtaps - 1) / 2` samples for a linear-phase FIR filter.
- `fir_filter_zero_phase(data, b, a=None)` → `filtered_signal`
  Zero-phase filtering (`scipy.signal.filtfilt`). No group delay, output
  stays time-aligned with input — use this for offline analysis/plotting.
- `kaiserord(ripple, width)` → `(numtaps, beta)`
  Thin wrapper around `scipy.signal.kaiserord`.

All cutoff/lowcut/highcut arguments are in the same units as `fs` (Hz by
default when `fs` is given explicitly), not normalized 0–1.

## signal_processing.transforms

### FFT and Spectral Analysis

- `fft(data, n=None)` → `spectrum`
- `ifft(spectrum, n=None)` → `signal`
- `fftfreq(n, d=1.0)` → `freqs`
- `fftshift(spectrum)` / `ifftshift(spectrum)`
- `power_spectral_density(data, fs=1.0, nperseg=None)` → `(freqs, psd)`
  Welch's method. Also available as `welch(...)` (alias).
- `spectrogram(data, fs=1.0, nperseg=256, noverlap=None)` → `(freqs, times, Sxx)`
- `stft(data, fs=1.0, nperseg=256, noverlap=None)` → `(freqs, times, Zxx)`
- `istft(Zxx, fs=1.0, nperseg=256, noverlap=None)` → `(t, x)`
- `coherence(signal1, signal2, fs=1.0, nperseg=256)` → `(freqs, Cxy)`
- `correlate(signal1, signal2, mode='full')` → `correlation`

## signal_processing.modulation

### Analog and Digital Modulation

- `am_modulate(message, carrier_freq, modulation_index=0.5, fs=1000.0)` → `modulated`
- `am_demodulate(am_signal, carrier_freq, fs=1000.0)` → `demodulated`
- `fm_modulate(message, carrier_freq, frequency_deviation, fs=1000.0)` → `modulated`
- `fm_demodulate(fm_signal, carrier_freq, frequency_deviation, fs=1000.0)` → `message`
- `bpsk_modulate(bits, carrier_freq, fs=1000.0, samples_per_symbol=16)` → `modulated`
  Also available as `psk_modulate(...)` (alias, kept for backward compatibility).
- `qpsk_modulate(bits, carrier_freq, fs=1000.0, samples_per_symbol=16)` → `modulated`
- `generate_pn_sequence(length, taps)` → `pn_sequence`
  LFSR-based pseudo-noise sequence, ±1 values. `taps` is 1-indexed
  (e.g. `[6, 1]` for the polynomial x⁶ + x + 1).

## signal_processing.utils

### Utility Functions

- `plot_signal(data, fs=1.0, title="Signal", xlabel="Time (s)", ylabel="Amplitude")`
- `plot_spectrum(data, fs=1.0, title="Frequency Spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude")`
- `plot_spectrogram(data, fs=1.0, title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")`
- `generate_test_signal(signal_type='chirp', duration=1.0, fs=1000.0, **kwargs)` → `signal`
  `signal_type` is one of `'sine'`, `'chirp'`, `'square'`, `'sawtooth'`, `'pulse'`, `'noise'`.
- `calculate_snr(data, noise=None)` → `snr_db`
  If `noise` is omitted, estimates a noise floor via high-pass filtering
  (a rough heuristic — see `doc/architecture.md` for its limitations).
- `eye_diagram(data, fs=1.0, symbol_rate=100.0, spans=2)`
