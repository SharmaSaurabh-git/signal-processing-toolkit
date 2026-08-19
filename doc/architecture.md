# Architecture Overview

## Package layout

```
signal_processing/
├── filters/        FIR filter design and application
│   └── fir.py
├── transforms/      Frequency-domain analysis
│   └── fft.py
├── modulation/       Analog & digital modulation/demodulation
│   └── digital.py
└── utils/            Signal generation, plotting, measurement
    └── plotting.py
```

Each subpackage is self-contained: `filters` only depends on `numpy`
and `scipy.signal`, `transforms` and `modulation` likewise. `utils`
is the one exception — `calculate_snr` optionally calls into
`filters.fir_highpass`/`fir_filter` to auto-estimate a noise floor
when no explicit noise signal is supplied.

The top-level `signal_processing/__init__.py` re-exports every public
function from all four subpackages into a single flat namespace, so
`import signal_processing as sp; sp.fir_lowpass(...)` works without
needing to know which submodule a function lives in. Each subpackage's
own `__init__.py` also re-exports its functions individually, so
`from signal_processing.filters import fir_lowpass` works too — useful
if you only need one part of the toolkit and want to avoid importing
matplotlib (pulled in by `utils`) for a headless/embedded use case.

## Design decisions

**Why `(b, a)` tuples instead of filter objects.** All filter design
functions return `(b, a)` coefficient arrays rather than a custom
`Filter` class, matching `scipy.signal`'s convention. This keeps the
toolkit interoperable with raw `scipy.signal` calls (`scipy.signal.freqz(b, a)`
works directly on the output) at the cost of not bundling filter
metadata (sample rate, filter type) alongside the coefficients.

**Causal vs. zero-phase filtering.** `fir_filter()` uses
`scipy.signal.lfilter`, a causal filter suitable for real-time/streaming
use, but it introduces a group delay of `(numtaps - 1) / 2` samples for
a linear-phase FIR filter. `fir_filter_zero_phase()` uses
`scipy.signal.filtfilt` (forward-backward filtering) to cancel that
delay for offline analysis, at roughly double the compute cost and at
the cost of no longer being causal. Pick the one that matches your use
case — see each function's docstring for details.

**No custom exceptions.** The toolkit raises standard `ValueError` for
invalid input (bad cutoff frequencies, mismatched array lengths, etc.)
rather than defining a custom exception hierarchy. For a library this
size, custom exception types add API surface without a corresponding
benefit — callers already know how to catch `ValueError`.

## Known limitations

- FIR filter design only supports the windowing method
  (`scipy.signal.firwin`). No IIR filter design (Butterworth,
  Chebyshev, elliptic) or Parks-McClellan (equiripple) FIR design is
  provided yet.
- `fm_demodulate` uses a simplified differentiate-the-phase approach
  rather than a true phase-locked loop, so it is more sensitive to
  noise than a PLL-based demodulator would be.
- `calculate_snr`'s auto-estimation mode (no explicit `noise` argument)
  assumes noise is concentrated above a fixed 0.1 (normalized) cutoff.
  This is a rough heuristic, not a rigorous noise estimator — pass an
  explicit `noise` array when you have one.
