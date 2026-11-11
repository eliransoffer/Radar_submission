"""Describe development OOF failures using saved predictions and NPZ metadata.

No training or inference. Only --d1-motion loads I/Q for an FFT diagnostic.
See research/README.md for the commands.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.make_splits import CLASSES, validate_assignment
from research.tracks import construct_tracks


def check_d1_motion(rows, summary, data_path, output):
    """Compare all D1 recordings without assigning flight/hover labels."""
    from research.spectral import doppler_spectra

    d1 = rows[rows.true_class == "D1"].copy()
    with np.load(data_path, allow_pickle=False) as data:
        # NPZ decoding reads the full segments member; retain only D1 cell 2.
        signals = data["segments"][d1.index, 2, :].astype(np.complex128)
    valid = np.isfinite(signals).all(axis=1) & np.any(signals != 0, axis=1)
    peaks = np.full(len(d1), np.nan)
    frequencies, power, _ = doppler_spectra(signals[valid])
    peaks[valid] = frequencies[np.argmax(power, axis=1)]
    d1["original_peak_hz"] = peaks
    d1.to_csv(output / "d1_motion_segments.csv", index_label="segment_index")

    groups = list(d1.groupby("recording_id"))
    fig, axes = plt.subplots(len(groups), 2, figsize=(10, 2 * len(groups)),
                             sharex="row", sharey="col", squeeze=False)
    statistics = []
    for row_number, (identifier, group) in enumerate(groups):
        group = group.sort_values("time_s")
        elapsed = group.time_s - group.time_s.iloc[0]
        # A whole-recording linear trend, not instantaneous target velocity.
        range_slope = np.polyfit(elapsed, group.range_m, 1)[0]
        statistics.append({
            "recording_id": identifier,
            "range_change_m": group.range_m.iloc[-1] - group.range_m.iloc[0],
            "range_slope_m_per_s": range_slope,
            "median_peak_hz": group.original_peak_hz.median(),
            "median_absolute_peak_hz": group.original_peak_hz.abs().median(),
            "invalid_peak_segments": group.original_peak_hz.isna().sum(),
        })
        for column, (value, label) in enumerate(
                (("range_m", "Range (m)"), ("original_peak_hz", "Original peak (Hz)"))):
            ax = axes[row_number, column]
            for correct, color, description in (
                    (True, "#009E73", "Correct D1 prediction"),
                    (False, "#D55E00", "Incorrect prediction")):
                mask = group.correct == correct
                ax.scatter(elapsed[mask], group.loc[mask, value], s=5,
                           color=color, label=description, rasterized=True)
            ax.set_ylabel(label)
            ax.set_xlabel("Time since recording start (s)")
            ax.grid(alpha=0.2)
            if column == 1:
                ax.axhline(0, color="0.5", linewidth=0.6)
            ax.set_title(f"D1 recording {identifier}: {group.correct.mean():.1%} correct",
                         fontweight="bold" if identifier in (5, 6) else "normal")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("D1 motion indicators and saved held-out CNN decisions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "d1_motion.png", dpi=300)
    fig.savefig(output / "d1_motion.pdf")
    plt.close(fig)
    result = summary.loc[[identifier for identifier, _ in groups]].join(
        pd.DataFrame(statistics).set_index("recording_id"))
    result.to_csv(output / "d1_motion_summary.csv")
    pd.crosstab(d1.recording_id, d1.predicted_class).reindex(
        columns=CLASSES, fill_value=0).to_csv(output / "d1_all_prediction_counts.csv")


def plot_range_accuracy(rows, output):
    """Fixed 20 m bins; saved whole-piece decisions scored on individual segments."""
    rows = rows.copy()
    edges = 20 * np.arange(int(rows.range_m.max() // 20) + 2)
    rows["range_bin"] = pd.cut(rows.range_m, edges, right=False)
    bins = rows.groupby(["true_class", "range_bin"], observed=True).agg(
        mean_range_m=("range_m", "mean"), accuracy=("correct", "mean"),
        n_segments=("correct", "size"), n_recordings=("recording_id", "nunique"))
    records = rows.groupby(
        ["true_class", "recording_id", "range_bin"], observed=True).agg(
            mean_range_m=("range_m", "mean"), accuracy=("correct", "mean"),
            n_segments=("correct", "size"))
    bins.to_csv(output / "accuracy_by_range.csv")
    records.to_csv(output / "accuracy_by_recording_range.csv")
    draw_range_accuracy(bins, output)


def draw_range_accuracy(bins, output, classes=CLASSES):
    """Plot existing bin summaries; optionally show only selected classes."""
    n_rows = (len(classes) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows + 1.7),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, classes):
        selected = bins.loc[name]
        positions = np.arange(len(selected))
        accuracy = 100 * selected.accuracy.to_numpy()
        bars = ax.bar(positions, accuracy, width=0.7, color="#3175a5")
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in accuracy],
                     padding=4, fontsize=9)
        labels = []
        for interval, row in selected.iterrows():
            lower = int(float(str(interval).split(",")[0].lstrip("[")))
            labels.append(f"{lower}-{lower + 20}\n"
                          f"{int(row.n_segments):,}\n{int(row.n_recordings)}")
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_title(name, fontsize=13, pad=12)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Distance bin (m) / segments / recordings", labelpad=10)
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    fig.supylabel("Segments classified correctly (%)", fontsize=11)
    fig.suptitle("Does classification accuracy change with target distance?\n"
                 "Selected classes | held-out development predictions | track-piece probability averaging",
                 fontsize=13)
    fig.text(0.5, 0.02, "Counts below each bar: distance bin, number of segments, number of recordings.\n"
             "Only occupied bins are shown. Recordings can appear in several bins; distance is not isolated from other conditions.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.13, 1, 0.85))
    fig.savefig(output / "accuracy_by_range.png", dpi=300)
    fig.savefig(output / "accuracy_by_range.pdf")
    plt.close(fig)


def analyze(args):
    if args.d1_motion and args.model != "cnn_peak_aligned":
        raise ValueError("The D1 motion plot currently compares saved CNN decisions only.")
    with np.load(args.data, allow_pickle=False) as data:
        rows = pd.DataFrame({
            "recording_id": data["recording_id"],
            "true_class": data["task_class"].astype(str),
            "time_s": data["time_s"],
            "range_m": data["range_m"],
        })
    rows["track_id"] = construct_tracks(rows.recording_id, rows.time_s)

    saved = pd.read_csv(args.predictions)
    representation = "five_cell_spectra" if args.model == "cnn_peak_aligned" else "G3"
    saved = saved[(saved.assignment == "development") &
                  (saved.model == args.model) & (saved.representation == representation) &
                  (saved.method == "whole_track")].copy()
    if (saved.segment_index.duplicated().any() or
            set(saved.segment_index) != set(rows.index)):
        raise ValueError("Expected exactly one development OOF prediction per segment.")
    saved = saved.set_index("segment_index").loc[rows.index]
    for name in ("recording_id", "true_class", "track_id"):
        if not np.array_equal(saved[name], rows[name]):
            raise ValueError(f"Saved predictions do not match dataset: {name}.")
    if not (saved.window.eq("rectangular").all() and
            saved.seed.eq(42).all() and saved.role.eq("development").all() and
            saved.predicted_class.isin(CLASSES).all()):
        raise ValueError("Expected selected rectangular development predictions.")

    split = json.loads(args.folds.read_text(encoding="utf-8"))
    folds = validate_assignment(split["runs"], rows.recording_id.to_numpy(),
                                rows.true_class.to_numpy())
    for fold, _, validation in folds:
        if not saved.iloc[validation].fold.eq(fold).all():
            raise ValueError("Saved predictions differ from development recording folds.")
    rows["fold"] = saved.fold.to_numpy()
    rows["predicted_class"] = saved.predicted_class.to_numpy()
    rows["correct"] = rows.true_class == rows.predicted_class
    if rows.groupby("track_id").predicted_class.nunique().max() != 1:
        raise ValueError("Whole-piece labels must be constant within each piece.")

    # Report both segment-weighted accuracy and equal-weight track-piece accuracy.
    pieces = rows.groupby("track_id", as_index=False).agg(
        recording_id=("recording_id", "first"),
        true_class=("true_class", "first"),
        predicted_class=("predicted_class", "first"),
        n_segments=("correct", "size"), correct=("correct", "first"))
    summary = rows.groupby("recording_id").agg(
        target_class=("true_class", "first"), fold=("fold", "first"),
        n_segments=("correct", "size"), segment_accuracy=("correct", "mean"),
        median_range_m=("range_m", "median"),
        min_range_m=("range_m", "min"), max_range_m=("range_m", "max"))
    summary["n_pieces"] = pieces.groupby("recording_id").size()
    summary["piece_accuracy"] = pieces.groupby("recording_id").correct.mean()

    # Only same-class recordings actually in this fold's training set are peers.
    comparisons = []
    for identifier, record in summary.iterrows():
        peers = summary[(summary.target_class == record.target_class) &
                        (summary.index != identifier)]
        training_peers = peers[peers.fold != record.fold]
        training_rows = rows[(rows.true_class == record.target_class) &
                             (rows.fold != record.fold)]
        comparisons.append({
            "recording_id": identifier,
            "same_class_other_recording_ids": ",".join(map(str, peers.index)),
            "same_class_other_median_min_m": peers.median_range_m.min(),
            "same_class_other_median_max_m": peers.median_range_m.max(),
            "training_recording_ids": ",".join(map(str, training_peers.index)),
            "training_median_min_m": training_peers.median_range_m.min(),
            "training_median_max_m": training_peers.median_range_m.max(),
            "training_segment_range_min_m": training_rows.range_m.min(),
            "training_segment_range_max_m": training_rows.range_m.max(),
        })
    summary = summary.join(pd.DataFrame(comparisons).set_index("recording_id"))

    # Counts show which labels replace D1; no flight/hover labels are inferred.
    d1 = rows[rows.recording_id.isin([5, 6])]
    d1_pieces = pieces[pieces.recording_id.isin([5, 6])]
    segment_counts = pd.crosstab(d1.recording_id, d1.predicted_class)
    piece_counts = pd.crosstab(d1_pieces.recording_id, d1_pieces.predicted_class)
    counts = pd.concat({"segments": segment_counts, "pieces": piece_counts},
                       names=["unit", "recording_id"])
    counts = counts.reindex(columns=CLASSES).fillna(0).astype(int)

    args.output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output / "recording_summary.csv")
    focus = summary.loc[[9, 18, 32, 68, 5, 6]]
    focus.to_csv(args.output / "focus_recordings.csv")
    counts.to_csv(args.output / "d1_prediction_counts.csv")
    plot_range_accuracy(rows, args.output)
    if args.d1_motion:
        check_d1_motion(rows, summary, args.data, args.output)
    print(focus[["target_class", "segment_accuracy", "piece_accuracy", "median_range_m"]])
    print(f"Saved {args.output}. Range associations do not establish error causes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--d1-motion", action="store_true",
                        help="Also plot all D1 recordings' range and original peak Doppler.")
    parser.add_argument("--model", choices=("cnn_peak_aligned", "logistic_regression", "xgboost"),
                        default="cnn_peak_aligned", help="Classical models use G3 (65 features).")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "runs/cnn_rectangular/validation_predictions.csv")
    parser.add_argument("--folds", type=Path,
                        default=ROOT / "research/recording_folds.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/recording_diagnostics")
    analyze(parser.parse_args())
