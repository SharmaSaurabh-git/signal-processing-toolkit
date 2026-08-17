# Signal Processing Toolkit
A comprehensive Python library for signal processing operations commonly used in Electrical and Computer Engineering.

## Features
- Digital filter design (FIR, IIR)
- Fourier transforms and spectral analysis
- Modulation/demodulation techniques
- Signal generation and analysis
- Visualization tools
- Educational examples and tutorials

## Installation
\`\`\`bash
pip install signal-processing-toolkit
\`\`\`

## Usage
\`\`\`python
import signal_processing as sp

# Design a low-pass filter
b, a = sp.fir_lowpass(cutoff=0.2, numtaps=51)

# Analyze a signal
freqs, psd = sp.welch(signal, fs=1000)
\`\`\`

## Contributing
Feel free to submit issues and pull requests!

## License
MIT
