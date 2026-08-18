# Signal Processing Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/SharmaSaurabh-git/signal-processing-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/SharmaSaurabh-git/signal-processing-toolkit/actions/workflows/ci.yml)
[![Code Size](https://img.shields.io/github/languages/code-size/SharmaSaurabh-git/signal-processing-toolkit)](https://github.com/SharmaSaurabh-git/signal-processing-toolkit)
[![GitHub Stars](https://img.shields.io/github/stars/SharmaSaurabh-git/signal-processing-toolkit?style=social)](https://github.com/SharmaSaurabh-git/signal-processing-toolkit/stargazers)

A comprehensive Python library for signal processing operations commonly used in Electrical and Computer Engineering. This toolkit implements fundamental algorithms for filtering, spectral analysis, modulation, and more - perfect for learning DSP concepts or prototyping signal processing systems.

## Features

### Core Modules
- **Filters**: FIR filter design (low-pass, high-pass, band-pass, band-stop) using windowing method
- **Transforms**: FFT/IFFT, power spectral density (Welch's method), spectrogram, STFT
- **Modulation**: AM/FM analog modulation, BPSK/QPSK digital modulation
- **Utilities**: Signal generation, plotting, SNR calculation, eye diagrams, PN sequences

### Key Capabilities
- Parameterizable filter design with customizable windows
- Real-time signal visualization tools
- Comprehensive test suite with GitHub Actions CI/CD
- Educational examples demonstrating core concepts
- Easy installation via pip (setup.py included)

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/SharmaSaurabh-git/signal-processing-toolkit.git
cd signal-processing-toolkit
pip install -e .

# Or install from PyPI (once published)
# pip install signal-processing-toolkit
```

## Quick Start

```python
import numpy as np
import signal_processing as sp

# Generate test signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2*np.pi*50*t) + 0.5*np.random.randn(1000)  # 50Hz sine + noise

# Design low-pass filter (cutoff = 100Hz)
b, a = sp.fir_lowpass(cutoff=100.0, numtaps=101, fs=1000.0)

# Filter signal
filtered = sp.fir_filter(signal, b, a)

# Analyze spectrum
freqs, psd = sp.welch(signal, fs=1000.0, nperseg=256)
filtered_freqs, filtered_psd = sp.welch(filtered, fs=1000.0, nperseg=256)

# Plot results (using built-in utilities)
sp.plot_signal(signal, fs=1000.0, title="Noisy Signal")
sp.plot_signal(filtered, fs=1000.0, title="Filtered Signal")
sp.plot_spectrum(signal, fs=1000.0, title="Signal Spectrum")
sp.plot_spectrum(filtered, fs=1000.0, title="Filtered Spectrum")
```

## Documentation

- [API Documentation](doc/api.md) - Detailed module references
- [Examples](signal_processing/examples/) - Ready-to-run Jupyter notebooks and scripts
- [Architecture Overview](doc/architecture.md) - Design decisions and implementation details
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

## Examples

See the [examples directory](signal_processing/examples/) for:
- Basic filtering demonstrations
- Modulation/demodulation examples
- Spectral analysis tutorials
- Comprehensive system demos

## Testing

Run the test suite with:
```bash
pytest
```

Tests are automatically run on every push and pull request via GitHub Actions.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by classic DSP textbooks and open-source implementations
- Built with NumPy, SciPy, and Matplotlib
- GitHub Actions for continuous integration
