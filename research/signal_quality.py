"""Contrast-quintile diagnostic using saved recording-held-out predictions only."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from radar_model.preprocessing import CLASSES
from research.spectral import FEATURE_LABELS, build_feature_matrix, doppler_spectra, extract_features
from research.feature_extraction import new_output
from research.tracks import construct_tracks
from research.make_splits import validate_assignment
MIN_SEGMENTS = 30
MIN_RECORDINGS = 3

def calculate_quality(segments, window="rectangular"):
    """Reuse the exact unaligned, central-cell feature and its 1e-30 safeguard."""
    features = build_feature_matrix(segments, window=window)
    ratio = features[:, list(FEATURE_LABELS).index("peak_background_ratio")]
    finite = np.isfinite(ratio) & (ratio > 0)
    contrast = np.full(len(ratio), np.nan)
    contrast[finite] = 10 * np.log10(ratio[finite])
    central_finite = np.isfinite(segments[:, 2, :]).all(axis=1)
    central_nonzero = np.any(segments[:, 2, :] != 0, axis=1)
    status = np.full(len(ratio), "valid", dtype=object)
    status[~finite] = "nonfinite_or_nonpositive_proxy"
    status[~central_nonzero] = "zero_central_cell"
    status[~central_finite] = "nonfinite_central_cell"
    # Verify the matrix wrapper against the existing direct spectrum/feature path.
    valid_indices = np.flatnonzero(finite)
    sample_step = max(1, int(finite.sum()) // 32)
    sample = valid_indices[::sample_step][:32]
    if len(sample):
        frequencies, _, power = doppler_spectra(segments[sample, 2, :].astype(np.complex128), window=window)
        direct = extract_features(frequencies, power, peak_half_width_hz=250)["peak_background_ratio"]
        np.testing.assert_allclose(ratio[sample], direct, rtol=1e-12, atol=0)
    return ratio, contrast, status

def contrast_bins(contrast):
    """Unlabeled full-development quantiles, before alignment; no class-specific bins."""
    valid = np.isfinite(contrast)
    if not valid.any():
        raise ValueError("No finite positive contrast values; cannot define bins.")
    cuts = np.quantile(contrast[valid], [0.2, 0.4, 0.6, 0.8], method="linear")
    np.testing.assert_array_equal(cuts, np.quantile(np.sort(contrast[valid]),
                                                [0.2, 0.4, 0.6, 0.8], method="linear"))
    bins = np.zeros(len(contrast), dtype=int)  # 0 explicitly denotes unavailable contrast.
    bins[valid] = np.searchsorted(cuts, contrast[valid], side="right") + 1
    if not np.all((bins[valid] >= 1) & (bins[valid] <= 5)):
        raise AssertionError("A finite segment failed exactly-one-bin assignment.")
    return cuts, bins

def group_metrics(rows, **identity):
    """Always show denominators; unavailable-contrast rows remain in separate groups."""
    counts = rows.true_class.value_counts()
    class_recordings = rows.groupby("true_class").recording_id.nunique()
    per_recording = rows.groupby(["true_class", "recording_id"]).correct.mean()
    balanced = per_recording.groupby(level=0).mean().mean()
    n = len(rows)
    recordings = rows.recording_id.nunique()
    support = "insufficient"
    if n >= MIN_SEGMENTS and recordings >= MIN_RECORDINGS:
        support = "supported"
    result = identity.copy()
    result.update({
        "n_segments": n,
        "n_recordings": recordings,
        "n_tracks": rows.track_id.nunique(),
        "n_classes": rows.true_class.nunique(),
        "n_valid_contrast": int(rows.contrast_db.notna().sum()),
        "accuracy": rows.correct.mean(),
        "class_balanced_recording_accuracy_present_classes": balanced,
        "support": support,
    })
    for name in CLASSES:
        result[f"n_{name}"] = int(counts.get(name, 0))
    for name in CLASSES:
        result[f"recordings_{name}"] = int(class_recordings.get(name, 0))
    return result


def analyze(args):
    """Reproduce selected-model contrast quintiles from complete saved OOF predictions."""
    if args.output.exists():
        raise FileExistsError("Choose a fresh quality-diagnostic directory under runs/.")
    with np.load(args.data, allow_pickle=False) as data:
        raw = data["segments"]
        metadata = pd.DataFrame({"segment_index": np.arange(len(raw)), "recording_id": data["recording_id"],
                                 "true_class": data["task_class"].astype(str), "time_s": data["time_s"]})
    metadata["track_id"] = construct_tracks(metadata.recording_id.to_numpy(), metadata.time_s.to_numpy())
    metadata["track_length"] = metadata.groupby("track_id").segment_index.transform("size")
    ratio, contrast, status = calculate_quality(raw, window=args.window)
    cuts, bins = contrast_bins(contrast)
    del raw
    metadata["peak_background_ratio"] = ratio
    metadata["contrast_db"] = contrast
    metadata["contrast_status"] = status
    metadata["quality_bin"] = bins
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[(predictions.assignment == args.assignment) &
                              (predictions.model == "cnn_peak_aligned") & (predictions.method == "whole_track")]
    if "window" in predictions:
        if not predictions.window.eq(args.window).all():
            raise ValueError("Prediction window differs from the quality diagnostic.")
    else:
        raise ValueError("Rectangular predictions must explicitly record their window.")
    if predictions.segment_index.duplicated().any() or set(predictions.segment_index) != set(metadata.segment_index):
        raise ValueError("Quality analysis requires one complete selected-model OOF prediction per observation.")
    predictions = predictions.set_index("segment_index").loc[metadata.segment_index]
    for name in ("recording_id", "true_class", "track_id"):
        if not np.array_equal(predictions[name], metadata[name]):
            raise ValueError(f"Prediction metadata differs from supplied data: {name}.")
    document = json.loads(args.folds.read_text(encoding="utf-8"))
    if "assignments" in document:
        items = document["assignments"]
    else:
        development = document.copy()
        development["assignment"] = "development"
        development["role"] = "development"
        items = [development]
    assignment = next(item for item in items if item["assignment"] == args.assignment)
    folds = validate_assignment(assignment["runs"], metadata.recording_id.to_numpy(), metadata.true_class.to_numpy())
    expected = np.empty(len(metadata), dtype=int)
    for fold, _, validation in folds:
        expected[validation] = fold
    if not (np.array_equal(predictions.fold, expected) and (predictions.seed == assignment["seed"]).all()
            and (predictions.role == assignment["role"]).all() and predictions.predicted_class.isin(CLASSES).all()):
        raise ValueError("Prediction fold/assignment provenance differs from saved recording splits.")
    metadata["predicted_class"] = predictions.predicted_class.to_numpy()
    metadata["correct"] = metadata.true_class == metadata.predicted_class
    identity = {"assignment": args.assignment, "role": assignment["role"], "window": args.window,
                "model": "cnn_peak_aligned_whole_track"}
    bin_rows, class_rows, record_rows = [], [], []
    for number in range(6):  # Bin 0 preserves observations with unavailable contrast.
        selected = metadata[metadata.quality_bin == number]
        bin_rows.append(group_metrics(selected, **identity, quality_bin=number))
        for name in CLASSES:
            class_rows.append(group_metrics(selected[selected.true_class == name], **identity,
                                            quality_bin=number, target_class=name))
    for identifier, selected in metadata.groupby("recording_id"):
        record_rows.append({**identity, "recording_id": int(identifier), "true_class": selected.true_class.iloc[0],
                            "n_segments": len(selected), "median_contrast_db": selected.contrast_db.median(),
                            "accuracy": selected.correct.mean()})
    output = new_output(args.output)
    metadata.to_csv(output / "segment_quality.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(output / "quality_bin_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "quality_by_class.csv", index=False)
    pd.DataFrame(record_rows).to_csv(output / "quality_by_recording.csv", index=False)
    (output / "configuration.json").write_text(json.dumps({"assignment": args.assignment,
        "window": args.window, "contrast_quintile_edges_db": cuts.tolist(), "unavailable_contrast_bin": 0,
        "peak_exclusion_hz": 250, "background_floor": 1e-30, "calibrated_snr": False,
        "note": "Unlabeled full-assignment quantiles are descriptive only, never training preprocessing. "
                "Class/recording composition changes across bins; no SNR-independence or causal conclusion."}, indent=2), encoding="utf-8")
    print(f"Saved {output}; no model was trained or run by this diagnostic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Supplied radar_segments.npz.")
    parser.add_argument("--predictions", type=Path, required=True, help="OOF validation_predictions.csv from research/evaluate.py cnn.")
    parser.add_argument("--folds", type=Path, default=ROOT / "research/repeated_folds.json")
    parser.add_argument("--assignment", default="development")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/signal_quality")
    parser.add_argument("--window", choices=("rectangular",), default="rectangular")
    analyze(parser.parse_args())
