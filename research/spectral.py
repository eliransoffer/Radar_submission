"""Original central-cell feature definitions used by the reported baselines.

Helpers only; feature_extraction.py and signal_quality.py provide entry points.
"""
import numpy as np
from radar_model.preprocessing import CLASSES, PRF, doppler_power

FEATURE_LABELS = {
    "peak_frequency_hz": "Peak Doppler (Hz)",
    "bandwidth_hz": "Power-weighted bandwidth (Hz)",
    "spectral_entropy": "Normalized spectral entropy",
    "peak_background_ratio": "Peak / median background (linear)",
    "off_peak_fraction": "Off-peak power fraction",
}

def doppler_spectra(signals, window="rectangular"):
    """Return frequencies (Hz), linear power, and unit-sum linear power.

    The same untapered FFT/power calculation is used by CNN processing.
    Dividing squared magnitude by 256 squared fixes the arbitrary power reference;
    this is a spectrum, not a calibrated power spectral density.
    """
    n_chirps = signals.shape[1]
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_chirps, d=1 / PRF))
    power = doppler_power(signals, window=window)
    total_power = power.sum(axis=1, keepdims=True)
    normalized_power = np.divide(
        power, total_power,
        out=np.full_like(power, np.nan), where=total_power > 0,
    )
    return frequencies, power, normalized_power

def extract_features(frequencies, normalized_power, peak_half_width_hz=250):
    """Five shape features, all calculated from linear spectral power.

    The excluded peak interval is +/-250 Hz by default (about four FFT bins
    on each side). It is a fixed exploratory definition, not a body-motion
    boundary. The peak/background ratio is a contrast proxy, NOT true SNR
    or scipy's topographic peak prominence. No background subtraction is used.
    """
    peak_indices = np.argmax(normalized_power, axis=1)
    peak_frequencies = frequencies[peak_indices]

    # Spectral centroid and standard deviation describe radial-motion spread.
    mean_frequency = np.sum(normalized_power * frequencies, axis=1)
    variance = np.sum(
        normalized_power * (frequencies - mean_frequency[:, None]) ** 2, axis=1
    )
    bandwidth = np.sqrt(np.maximum(variance, 0))

    # Entropy approaches 0 for concentrated energy and 1 for uniform energy.
    entropy = -np.sum(
        normalized_power * np.log(np.maximum(normalized_power, 1e-30)), axis=1
    ) / np.log(len(frequencies))

    outside_peak = (
        np.abs(frequencies[None, :] - peak_frequencies[:, None]) > peak_half_width_hz
    )
    background = np.nanmedian(
        np.where(outside_peak, normalized_power, np.nan), axis=1
    )
    peak_power = np.max(normalized_power, axis=1)
    peak_background_ratio = peak_power / np.maximum(background, 1e-30)
    off_peak_fraction = np.sum(normalized_power * outside_peak, axis=1)

    return {
        "peak_frequency_hz": peak_frequencies,
        "bandwidth_hz": bandwidth,
        "spectral_entropy": entropy,
        "peak_background_ratio": peak_background_ratio,
        "off_peak_fraction": off_peak_fraction,
    }

def build_feature_matrix(segments, window="rectangular"):
    """Return the existing five features in FEATURE_LABELS order.

    Input: (N, 5, 256) complex segments, as supplied to the required predict().
    Only middle range cell 2 is used. No labels, recording IDs, track context,
    or learned statistics enter extraction. The peak half-width stays 250 Hz.
    Zero/non-finite signals produce missing features; a training-fitted imputer
    handles them later without dropping rows or changing prediction order.
    """
    feature_names = list(FEATURE_LABELS)
    matrix = np.full((len(segments), len(feature_names)), np.nan)
    for start in range(0, len(segments), 2048):
        stop = min(start + 2048, len(segments))
        signals = segments[start:stop, 2, :].astype(np.complex128)
        valid = np.isfinite(signals).all(axis=1) & np.any(signals != 0, axis=1)
        if not valid.any():
            continue
        frequencies, _, normalized_power = doppler_spectra(signals[valid], window=window)
        features = extract_features(frequencies, normalized_power, peak_half_width_hz=250)
        indices = np.arange(start, stop)[valid]
        for column, name in enumerate(feature_names):
            matrix[indices, column] = features[name]
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix
