# API Documentation

## signal_processing.filters
### FIR Filter Design
- `fir_lowpass(cutoff, numtaps, window='hamming', fs=2.0)` → (b, a)
- `fir_highpass(cutoff, numtaps, window='hamming', fs=2.0)` → (b, a)
- `fir_bandpass(lowcut, highcut, numtaps, window='hamming', fs=2.0)` → (b, a)
- `fir_bandstop(lowcut, highcut, numtaps, window='hamming', fs=2.0)` → (b, a)
- `fir_filter(signal, b, a=None)` → filtered_signal

## signal_processing.transforms
### FFT and Spectral Analysis
- `fft(signal, n=None)` → spectrum
- `ifft(spectrum, n=None)` → signal
- `welch(signal, fs=1.0, nperseg=None)` → (freqs, psd)
- `spectrogram(signal, fs=1.0, nperseg=256, noverlap=None)` → (freqs, times, Sxx)
- `stft(signal, fs=1.0, nperseg=256, noverlap=None)` → (freqs, times, Zxx)

## signal_processing.modulation
### Analog and Digital Modulation
- `am_modulate(message, carrier_freq, modulation_index=0.5, fs=1000.0)` → modulated
- `am_demodulate(am_signal, carrier_freq, fs=1000.0)` → demodulated
- `fm_modulate(message, carrier_freq, frequency_deviation, fs=1000.0)` → modulated
- `fm_demodulate(fm_signal, carrier_freq, frequency_deviation, fs=1000.0)` → message
- `bpsk_modulate(bits, carrier_freq, fs=1000.0, samples_per_symbol=16)` → modulated
- `qpsk_modulate(bits, carrier_freq, fs=1000.0, samples_per_symbol=16)` → modulated

## signal_processing.utils
### Utility Functions
- `plot_signal(signal, fs=1.0, title="Signal", xlabel="Time (s)", ylabel="Amplitude")`
- `plot_spectrum(signal, fs=1.0, title="Frequency Spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude")`
- `plot_spectrogram(signal, fs=1.0, title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")`
- `generate_test_signal(signal_type='chirp', duration=1.0, fs=1000.0, **kwargs)` → signal
- `calculate_snr(signal, noise=None)` → snr_db
- `eye_diagram(signal, fs=1.0, symbol_rate=100.0, spans=2)`
- `generate_pn_sequence(length, taps)` → pn_sequence
