"""Reported 5/9/17/25/57/65 segment features; original FFT-backend formulas.

Run the extraction entry point below with only the supplied radar_segments.npz.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from research.spectral import PRF, FEATURE_LABELS, doppler_spectra, extract_features
from research.tracks import construct_tracks

ORIGINAL = tuple(FEATURE_LABELS)
SHAPE = ("spectral_asymmetry", "secondary_peak_ratio", "near_band_fraction", "far_band_fraction")
FRACTIONS = tuple(f"cell{c}_power_fraction" for c in range(5))
SIMILARITY = tuple(f"cell{c}_cosine" for c in (0, 1, 3, 4))
SPATIAL = FRACTIONS[:4] + SIMILARITY
SEGMENT_NAMES = ORIGINAL + SHAPE + FRACTIONS + SIMILARITY
# Keep all five power fractions in diagnostics; the 17 predictors omit cell 4.
BASE_NAMES = ORIGINAL + SHAPE + SPATIAL
BANDS = ("far_negative", "mid_negative", "near_negative", "core_negative",
         "core_positive", "near_positive", "mid_positive", "far_positive")
EDGES = np.array([-32, -8, -2, 0, 3, 9, 33])
MIN_BAND_FRACTION = 1e-8
ENERGY_NAMES = tuple(f"band_energy_{band}" for band in BANDS)
SPATIAL_NAMES = tuple(f"cell{cell}_fraction_{band}" for band in BANDS for cell in range(4))
ENTROPY_NAMES = tuple(f"spatial_entropy_{band}" for band in BANDS)
FEATURE_NAMES = BASE_NAMES + ENERGY_NAMES + SPATIAL_NAMES + ENTROPY_NAMES
SIZES = (5, 9, 17, 25, 57, 65)

def segment_features(segments, window="rectangular"):
    """Return measured features, unnormalized cell powers and validity flags."""
    result = np.full((len(segments), len(SEGMENT_NAMES)), np.nan)
    cell_energy = np.full((len(segments), 5), np.nan)
    invalid_cells = ~np.isfinite(segments).all(axis=2)
    for start in range(0, len(segments), 1024):
        stop = min(start + 1024, len(segments))
        signals = segments[start:stop].astype(np.complex128)
        invalid = invalid_cells[start:stop]
        signals[invalid] = 0  # Prevent invalid cells contaminating FFT arithmetic.
        flattened = signals.reshape(-1, 256)
        frequencies, power, normalized = doppler_spectra(flattened, window=window)
        power = power.reshape(-1, 5, 256)
        normalized = normalized.reshape(-1, 5, 256)
        power[invalid] = np.nan
        normalized[invalid] = np.nan
        energy = power.sum(axis=2)
        cell_energy[start:stop] = energy
        central = normalized[:, 2, :]
        valid = np.isfinite(central).all(axis=1)
        base = extract_features(frequencies, central[valid], peak_half_width_hz=250)
        rows = np.arange(start, stop)[valid]
        for column, name in enumerate(ORIGINAL):
            result[rows, column] = base[name]
        # No shifting: compare measured bins by signed distance from the peak.
        frequency_spacing = PRF / 256
        for local in np.flatnonzero(valid):
            central_power = central[local]
            peak = int(np.argmax(central_power))
            offset = frequencies - frequencies[peak]
            asymmetry = central_power[offset > 0].sum() - central_power[offset < 0].sum()
            peaks, _ = find_peaks(central_power)
            peak_separation = np.abs(frequencies[peaks] - frequencies[peak])
            peaks = peaks[peak_separation >= 4 * frequency_spacing]
            secondary = 0.0
            if len(peaks):
                secondary = central_power[peaks].max() / central_power[peak]
            near_bins = (np.abs(offset) > 2 * frequency_spacing) & (np.abs(offset) <= 8 * frequency_spacing)
            near = central_power[near_bins].sum()
            far = central_power[np.abs(offset) > 8 * frequency_spacing].sum()
            result[start + local, len(ORIGINAL):len(ORIGINAL) + len(SHAPE)] = [asymmetry, secondary, near, far]
        # Fractions use unnormalized power, never independently normalized spectra.
        total = energy.sum(axis=1, keepdims=True)
        relative = np.divide(energy, total, out=np.full_like(energy, np.nan),
                             where=np.isfinite(total) & (total > 0))
        offset = len(ORIGINAL) + len(SHAPE)
        result[start:stop, offset:offset + 5] = relative
        for j, cell in enumerate((0, 1, 3, 4)):
            neighbor = normalized[:, cell, :]
            numerator = (neighbor * central).sum(axis=1)
            denominator = np.sqrt((neighbor**2).sum(axis=1) * (central**2).sum(axis=1))
            cosine = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                               where=np.isfinite(denominator) & (denominator > 0))
            result[start:stop, offset + 5 + j] = np.clip(cosine, 0, 1)
    result[~np.isfinite(result)] = np.nan
    return result, cell_energy, invalid_cells

def power_matrix(segments, window="rectangular"):
    """Only I/Q enters processing; invalid segments are flagged, not repaired as data."""
    signals = segments.astype(np.complex128)
    valid_input = np.isfinite(signals).all(axis=(1, 2))
    signals[~valid_input] = 0
    flattened = signals.reshape(-1, 256)
    frequencies, power, _ = doppler_spectra(flattened, window=window)
    return frequencies, power.reshape(-1, 5, 256), valid_input

def from_power(power, valid_input=None):
    """Convert (N,5,256) linear power to 48 global features and diagnostics.

    G1 has 8 band-power fractions, G2 has 32 within-band cell fractions, and
    G3 has 8 spatial entropies. Each segment is processed independently.
    """
    finite = np.isfinite(power).all(axis=(1, 2)) & (power >= 0).all(axis=(1, 2))
    if valid_input is not None:
        finite &= valid_input
    clean = np.where(np.isfinite(power), power, 0)
    total = clean.sum(axis=(1, 2))
    central = clean[:, 2, :].sum(axis=1)
    valid = finite & np.isfinite(total) & (total > 0) & (central > 0)
    peaks = np.argmax(clean[:, 2, :], axis=1)  # First maximum, as in the existing code.
    offsets = np.arange(256)[None, :] - peaks[:, None]
    # Map every peak-relative frequency bin to one of the eight fixed bands.
    membership = np.digitize(offsets, EDGES, right=False)
    # Shape: segment x band x frequency. Each bin belongs to exactly one band.
    masks = membership[:, None, :] == np.arange(8)[None, :, None]
    np.testing.assert_array_equal(masks.sum(axis=1), np.ones_like(offsets))
    counts = masks.sum(axis=2)
    # Normalize globally, not per cell: cross-cell energy ratios remain available.
    normalized = np.divide(clean, total[:, None, None], out=np.full_like(clean, np.nan),
                           where=valid[:, None, None])
    # Sum frequency bins separately for every band, preserving all five cells.
    band_energies = []
    for band in range(8):
        energy_in_band = (normalized * masks[:, band, None, :]).sum(axis=2)
        band_energies.append(energy_in_band)
    cell_band_energy = np.stack(band_energies, axis=1)  # (N, band, cell)
    raw_energy = cell_band_energy.sum(axis=2)
    reliable = valid[:, None] & (counts > 0) & (raw_energy > MIN_BAND_FRACTION)
    spatial = np.divide(cell_band_energy, raw_energy[:, :, None], out=np.full_like(cell_band_energy, np.nan),
                        where=reliable[:, :, None])
    energy = np.where(reliable, raw_energy, np.nan)
    entropy = -np.sum(spatial * np.log(np.maximum(spatial, 1e-30)), axis=2) / np.log(5)
    entropy = np.clip(entropy, 0, 1)
    # Four independent cell fractions per band; cell 4 is retained in diagnostics.
    g1_features = energy                         # 8 band-power fractions.
    g2_features = spatial[:, :, :4].reshape(-1, 32)  # 8 bands x 4 cell fractions.
    g3_features = entropy                        # 8 spatial entropies.
    features = np.column_stack([g1_features, g2_features, g3_features])
    if valid.any():
        np.testing.assert_allclose(raw_energy[valid].sum(axis=1), 1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.nansum(energy[valid], axis=1), 1, rtol=0,
                                   atol=8*MIN_BAND_FRACTION + 1e-12)
    if reliable.any():
        np.testing.assert_allclose(spatial[reliable].sum(axis=1), 1, rtol=1e-12, atol=1e-12)
    if np.isinf(features).any():
        raise AssertionError("Infinity must never enter the feature matrix.")
    diagnostics = {"raw_energy": raw_energy, "spatial": spatial, "counts": counts,
                   "reliable": reliable, "valid": valid, "finite_input": finite, "peaks": peaks}
    return features, diagnostics


def new_output(path):
    """Create a fresh research output under runs/, protecting submitted artifacts."""
    path = Path(path).resolve()
    runs = ROOT / "runs"
    if not path.is_relative_to(runs) or path == runs:
        raise ValueError("Research outputs must be subdirectories of the repository's runs/.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def extract(args):
    """Extract the fixed segment-local features; no imputation, labels or fitting enter them."""
    if args.output.exists():
        raise FileExistsError("Choose a fresh feature-cache directory.")
    with np.load(args.data, allow_pickle=False) as data:
        raw = data["segments"]
        metadata = pd.DataFrame({"segment_index": np.arange(len(raw)), "recording_id": data["recording_id"],
                                 "true_class": data["task_class"].astype(str), "time_s": data["time_s"]})
    metadata["track_id"] = construct_tracks(metadata.recording_id.to_numpy(), metadata.time_s.to_numpy())
    print("Extracting original central/spatial features...", flush=True)
    segment, _, _ = segment_features(raw, window=args.window)
    base_columns = []
    for name in BASE_NAMES:
        base_columns.append(SEGMENT_NAMES.index(name))
    base = segment[:, base_columns]
    bands = np.full((len(raw), 48), np.nan)
    for start in range(0, len(raw), 1024):
        stop = min(start + 1024, len(raw))
        _, power, valid = power_matrix(raw[start:stop], window=args.window)
        bands[start:stop], _ = from_power(power, valid)
    features = np.column_stack([base, bands])
    output = new_output(args.output)
    np.savez_compressed(output / "features.npz", features=features, segment_index=metadata.segment_index.to_numpy())
    metadata.to_csv(output / "segments.csv", index=False)
    (output / "configuration.json").write_text(json.dumps({"feature_names": list(FEATURE_NAMES),
        "backend": "fft", "window": args.window, "sizes": SIZES, "band_edges": EDGES.tolist(),
        "minimum_band_fraction": MIN_BAND_FRACTION, "learned_preprocessing": False}, indent=2), encoding="utf-8")
    print(f"Saved {output}. Imputation/scaling belong inside training folds, not extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Supplied radar_segments.npz only.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/features")
    parser.add_argument("--window", choices=("rectangular",), default="rectangular")
    extract(parser.parse_args())
