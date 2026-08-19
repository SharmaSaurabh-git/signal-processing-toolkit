"""
Package-level smoke tests: verifies the public `signal_processing`
namespace imports cleanly and re-exports the documented top-level API.
Function-specific behavior is covered in test_filters.py,
test_transforms.py, test_modulation.py, and test_utils.py.
"""
import signal_processing as sp


def test_import_and_version():
    assert sp.__version__ == "0.2.0"


def test_top_level_namespace_exports_documented_functions():
    """Every function name in doc/api.md should be reachable as sp.<name>."""
    expected = [
        "fir_lowpass", "fir_highpass", "fir_bandpass", "fir_bandstop", "fir_filter",
        "fft", "ifft", "power_spectral_density", "welch", "spectrogram", "stft",
        "am_modulate", "am_demodulate", "fm_modulate", "fm_demodulate",
        "bpsk_modulate", "qpsk_modulate",
        "plot_signal", "plot_spectrum", "plot_spectrogram",
        "generate_test_signal", "calculate_snr", "eye_diagram", "generate_pn_sequence",
    ]
    missing = [name for name in expected if not hasattr(sp, name)]
    assert missing == [], f"Documented functions missing from top-level namespace: {missing}"
