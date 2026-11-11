"""Compare saved range diagnostics; no training, inference or raw data loading.

First run recording_diagnostics.py separately for CNN, LR and XGBoost.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare(args):
    names = ("CNN", "LR (65 features)", "XGBoost (65 features)")
    tables, recordings = [], []
    for name, folder in zip(names, (args.cnn, args.lr, args.xgb)):
        table = pd.read_csv(folder / "accuracy_by_range.csv")
        table = table.set_index(["true_class", "range_bin"])
        if tables:
            # Identical bins, sample counts and ranges are required for comparison.
            columns = ["mean_range_m", "n_segments", "n_recordings"]
            pd.testing.assert_frame_equal(tables[0][columns], table[columns])
        tables.append(table)
        recordings.append(pd.read_csv(folder / "recording_summary.csv"))
    args.output.mkdir(parents=True, exist_ok=False)
    combined = pd.concat(tables, keys=names, names=["model"])
    combined.to_csv(args.output / "accuracy_by_range.csv")
    pd.concat(recordings, keys=names, names=["model"]).to_csv(
        args.output / "recording_summary.csv", index_label=["model", "row"])

    classes = ("D6", "human", "D1")
    colors = ("#0072B2", "#E69F00", "#009E73")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, target in zip(axes, classes):
        positions = np.arange(len(tables[0].loc[target]))
        for number, (name, table, color) in enumerate(zip(names, tables, colors)):
            selected = table.loc[target]
            accuracy = 100 * selected.accuracy.to_numpy()
            ax.bar(positions + (number - 1) * 0.25, accuracy,
                   width=0.24, color=color, label=name)
        labels = []
        for interval, values in tables[0].loc[target].iterrows():
            lower = int(float(interval.split(",")[0].lstrip("[")))
            labels.append(f"{lower}-{lower + 20}\n"
                          f"{int(values.n_segments):,} / {int(values.n_recordings)}")
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"True class: {target}", fontsize=12)
        ax.set_xlabel("Distance bin (m)", labelpad=10)
    axes[0].set_ylabel("Segments classified correctly (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89),
               ncol=3, frameon=False)
    fig.suptitle("Held-out accuracy by target distance\n"
                 "All models use whole-piece probability averaging (at most 30 segments)",
                 fontsize=13)
    fig.text(0.5, 0.02, "Counts below each distance bin: segments / recordings, identical for all models.\n"
             "Development recording-held-out predictions; range associations do not establish causation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.81))
    fig.savefig(args.output / "range_model_comparison.png", dpi=300)
    fig.savefig(args.output / "range_model_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--lr", type=Path, required=True)
    parser.add_argument("--xgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())
