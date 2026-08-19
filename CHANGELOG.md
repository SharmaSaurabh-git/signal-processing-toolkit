# Changelog

All notable changes to this project are documented here.

## [0.2.0] — Correctness and stability fixes

### Fixed
- **Critical:** `fir_filter`, `power_spectral_density`, `spectrogram`, `stft`,
  `plot_spectrogram`, and `calculate_snr` all crashed with `AttributeError`
  when called. Cause: each function had a parameter named `signal`/`data`
  that shadowed the module-level `import scipy.signal as signal`, so calls
  like `signal.lfilter(...)` inside the function resolved to the array
  argument instead of the scipy module. Renamed the import alias to `sps`
  throughout and audited every function for the same pattern.
- **Critical:** `fir_lowpass`, `fir_highpass`, `fir_bandpass`, and
  `fir_bandstop` were double-normalizing the cutoff frequency (dividing by
  Nyquist, then also passing `fs=fs` to `scipy.signal.firwin`, which expects
  cutoff in Hz when `fs` is given). A filter requested with a 100 Hz cutoff
  at a 1000 Hz sample rate actually cut off around 7 Hz. Fixed by passing
  the cutoff directly in Hz.
- `calculate_snr(noise=None)` additionally had a broken relative import
  (`from .filters import fir_highpass`, which doesn't exist from
  `utils/`) on top of the shadowing bug above.
- `eye_diagram` plotted `eye_matrix.T` against `time_axis`, a dimension
  mismatch that raised a `ValueError` from matplotlib. The array should
  not have been transposed.
- `signal_processing/examples/basic_filtering.py` imported
  `fir_lowpass`/`plot_signal` from package `__init__.py` files that never
  re-exported them — every subpackage `__init__.py` (`filters`,
  `transforms`, `modulation`, `utils`) now properly imports and exposes
  its public functions.
- `demo/generate_demo_plots.py` and
  `signal_processing/examples/comprehensive_demo.py` contained literal
  unescaped newlines inside string literals — both files raised
  `SyntaxError` and could not be imported or run at all.
- Both files also called `sp.welch(...)` and `sp.chirp(...)`, neither of
  which existed in the public API (the real names were
  `power_spectral_density` and `generate_test_signal('chirp', ...)`).

### Added
- `fir_filter_zero_phase()` — a zero-phase (forward-backward) filtering
  option using `scipy.signal.filtfilt`, for cases where output must stay
  time-aligned with input (offline analysis, plotting). Documented the
  group-delay behavior of the existing causal `fir_filter` so the
  difference is clear.
- Input validation with clear error messages across filter design,
  modulation, and utility functions (invalid cutoff frequencies, empty
  bit arrays, mismatched signal/noise lengths, invalid LFSR taps, etc.).
- `welch` as an explicit alias for `power_spectral_density`, matching
  both the documentation and common DSP terminology.
- `bpsk_modulate` as the primary name for binary PSK modulation, matching
  `doc/api.md`; `psk_modulate` kept as an alias for backward compatibility.
- Full test suite covering every public function, including the ones
  above that previously had zero coverage (`tests/test_filters.py`,
  `tests/test_transforms.py`, `tests/test_modulation.py`,
  `tests/test_utils.py`). 46 tests total.
- `.gitignore` for build artifacts, caches, and generated demo output.
- This changelog.

### Removed
- `signal_processing/tests/` — an empty, unused duplicate test package
  that shipped inside the library itself alongside the real `tests/`
  directory at the project root.
- Build artifacts (`__pycache__`, `*.egg-info`) that had been committed
  to the repository.

## [0.1.0] — Initial release

- FIR filter design (low-pass, high-pass, band-pass, band-stop).
- FFT/IFFT, power spectral density, spectrogram, STFT.
- AM/FM analog modulation, BPSK/QPSK digital modulation.
- Signal generation, plotting, SNR calculation, eye diagrams, PN sequences.
