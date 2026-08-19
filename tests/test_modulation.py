"""Tests for signal_processing.modulation (AM/FM/BPSK/QPSK/PN sequences)."""
import numpy as np
import pytest

from signal_processing.modulation import (
    am_modulate, am_demodulate,
    fm_modulate, fm_demodulate,
    bpsk_modulate, psk_modulate, qpsk_modulate,
    generate_pn_sequence,
)


def test_am_modulate_demodulate_recovers_message():
    fs = 1000.0
    t = np.arange(0, 1, 1 / fs)
    message = np.sin(2 * np.pi * 5 * t)
    modulated = am_modulate(message, carrier_freq=50.0, modulation_index=0.5, fs=fs)
    demodulated = am_demodulate(modulated, carrier_freq=50.0, fs=fs)
    correlation = np.correlate(message[:500], demodulated[:500], mode="valid")
    assert np.max(correlation) > 100


def test_fm_modulate_demodulate_recovers_message():
    fs = 2000.0
    t = np.arange(0, 1, 1 / fs)
    message = np.sin(2 * np.pi * 5 * t)
    modulated = fm_modulate(message, carrier_freq=200.0, frequency_deviation=50.0, fs=fs)
    demodulated = fm_demodulate(modulated, carrier_freq=200.0, frequency_deviation=50.0, fs=fs)
    correlation = np.corrcoef(message[100:900], demodulated[100:900])[0, 1]
    assert correlation > 0.8


def test_bpsk_modulate_and_psk_alias_match():
    bits = np.array([0, 1, 1, 0, 1])
    a = bpsk_modulate(bits, carrier_freq=50.0, fs=1000.0, samples_per_symbol=16)
    b = psk_modulate(bits, carrier_freq=50.0, fs=1000.0, samples_per_symbol=16)
    assert np.array_equal(a, b)
    assert len(a) == len(bits) * 16


def test_bpsk_modulate_rejects_empty_input():
    with pytest.raises(ValueError):
        bpsk_modulate(np.array([]), carrier_freq=50.0, fs=1000.0)


def test_qpsk_modulate_runs_with_even_and_odd_bits():
    even_bits = np.array([0, 1, 1, 0])
    odd_bits = np.array([0, 1, 1])
    out_even = qpsk_modulate(even_bits, carrier_freq=50.0, fs=1000.0, samples_per_symbol=8)
    out_odd = qpsk_modulate(odd_bits, carrier_freq=50.0, fs=1000.0, samples_per_symbol=8)
    assert len(out_even) == 2 * 8  # 4 bits -> 2 symbols
    assert len(out_odd) == 2 * 8   # 3 bits padded to 4 -> 2 symbols


def test_generate_pn_sequence_length_and_values():
    """Regression test: this function was documented (doc/api.md) but had
    zero test coverage before this suite."""
    seq = generate_pn_sequence(length=63, taps=[6, 1])
    assert len(seq) == 63
    assert set(np.unique(seq)).issubset({-1, 1})


def test_generate_pn_sequence_rejects_bad_input():
    with pytest.raises(ValueError):
        generate_pn_sequence(length=0, taps=[6, 1])
    with pytest.raises(ValueError):
        generate_pn_sequence(length=10, taps=[])
