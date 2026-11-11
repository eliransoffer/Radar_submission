"""Untapered Doppler FFT and peak alignment, shared by training and inference."""

import numpy as np

CLASSES = ("D1", "D2", "D3", "D4", "D5", "D6", "human", "bird", "reflector")
PRF = 17_000


def doppler_power(signals, window="rectangular"):
    """Convert flattened cell signals (M,256) to shifted linear Doppler power."""
    if window != "rectangular":
        raise ValueError("Only rectangular (no taper) preprocessing is supported.")
    # Chirp-to-chirp phase evolution encodes Doppler. No taper is applied.
    coefficients = np.fft.fft(signals, axis=1)
    coefficients = np.fft.fftshift(coefficients, axes=1)
    power = np.abs(coefficients) ** 2
    return power / 256**2


def shift_without_wrap(power, shift_bins):
    """Shift all five cell spectra equally; truncate boundary energy and zero-pad."""
    shifted = np.zeros_like(power)
    if shift_bins >= 0:
        shifted[:, shift_bins:] = power[:, :256 - shift_bins]
    else:
        shifted[:, :256 + shift_bins] = power[:, -shift_bins:]
    return shifted


def build_aligned_inputs(segments, window="rectangular"):
    """Return float32 (N,5,256) inputs, matching the evaluated transform exactly."""
    inputs = np.empty((len(segments), 5, 256), dtype=np.float32)
    for start in range(0, len(segments), 1024):
        stop = min(start + 1024, len(segments))
        signals = segments[start:stop].astype(np.complex128)
        valid_segments = np.isfinite(signals).all(axis=(1, 2))
        signals[~valid_segments] = 0

        # Treat every cell as one row for the chirp-axis FFT, then restore cells.
        cell_signals = signals.reshape(-1, 256)
        power = doppler_power(cell_signals, window=window)
        power = power.reshape(-1, 5, 256)

        central_power = power[:, 2, :]
        has_peak = central_power.sum(axis=1) > 0
        peak_bins = np.argmax(central_power, axis=1)
        shifts = np.where(has_peak, 128 - peak_bins, 0)
        aligned_power = np.empty_like(power)
        for row, shift in enumerate(shifts):
            # The dominant component is a reference, not necessarily body velocity.
            # No circular roll: energy leaving the sampled interval is discarded.
            aligned_power[row] = shift_without_wrap(power[row], int(shift))

        total_power = aligned_power.sum(axis=(1, 2), keepdims=True)
        # A single five-cell denominator preserves relative cell powers, not SNR invariance.
        fractions = np.divide(
            aligned_power, total_power,
            out=np.zeros_like(aligned_power), where=total_power > 0,
        )
        log_power = 10 * np.log10(np.maximum(fractions, 1e-8))
        inputs[start:stop] = (log_power / 80).astype(np.float32)
    return inputs
