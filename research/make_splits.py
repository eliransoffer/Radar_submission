"""Reproduce stored recording-grouped assignments using the supplied dataset.

Run from the repository root: python research/make_splits.py --data data/radar_segments.npz
Writes fresh JSON under runs/; --repeated adds fixed-seed robustness assignments.
Those assignments reuse recordings and are not independent test sets.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np


N_FOLDS = 3
CLASSES = ("D1", "D2", "D3", "D4", "D5", "D6", "human", "bird", "reflector")
ROOT = Path(__file__).resolve().parents[1]


def validate_assignment(runs, recording_ids, classes):
    """Check recording isolation, class coverage, and exactly-once validation."""
    if len(runs) != 3 or len({run["fold"] for run in runs}) != 3:
        raise ValueError("Expected three distinct folds per assignment.")
    all_ids = set(recording_ids.tolist())
    seen_validation = []
    folds = []
    for run in runs:
        train_ids = set(run["train_recording_ids"])
        validation_ids = set(run["validation_recording_ids"])
        if train_ids & validation_ids or (train_ids | validation_ids) != all_ids:
            raise ValueError("Overlapping, missing, or unknown recordings in splits.")
        train = np.flatnonzero(np.isin(recording_ids, sorted(train_ids)))
        validation = np.flatnonzero(np.isin(recording_ids, sorted(validation_ids)))
        for indices in (train, validation):
            if set(classes[indices]) != set(CLASSES):
                raise ValueError("Every training and validation fold must contain all classes.")
        folds.append((run["fold"], train, validation))
        seen_validation.extend(validation_ids)
    if set(seen_validation) != all_ids or len(seen_validation) != len(all_ids):
        raise ValueError("Each recording must validate exactly once per assignment.")
    return folds


def make_splits(data_path, output_path, seed, shuffle_recordings=False):
    """Apply the original class-aware recording allocation and optionally save its JSON."""
    # Only read metadata; the large signal array is not needed.
    with np.load(data_path, allow_pickle=False) as data:
        recording_ids = data["recording_id"]
        classes = data["task_class"]

    if recording_ids.ndim != 1 or classes.shape != recording_ids.shape:
        raise ValueError("recording_id and task_class must be matching 1D arrays.")
    if set(np.unique(classes)) != set(CLASSES):
        raise ValueError("Expected exactly the nine assignment classes.")

    # Summarize each recording: its class and number of segments.
    recordings = []
    for recording_id in np.unique(recording_ids):
        mask = recording_ids == recording_id
        labels = np.unique(classes[mask])
        if len(labels) != 1:
            raise ValueError(f"Recording {recording_id} contains multiple classes.")
        recordings.append({
            "id": int(recording_id),
            "class": str(labels[0]),
            "segments": int(mask.sum()),
        })

    rng = random.Random(seed)
    folds = [[] for _ in range(N_FOLDS)]

    for class_name in CLASSES:
        class_recordings = [r for r in recordings if r["class"] == class_name]
        if len(class_recordings) < N_FOLDS:
            raise ValueError(f"{class_name} needs at least {N_FOLDS} recordings.")

        # Shuffle first so equal-sized recordings have reproducible random order.
        rng.shuffle(class_recordings)
        if not shuffle_recordings:
            class_recordings.sort(key=lambda r: r["segments"], reverse=True)
        segment_counts = [0] * N_FOLDS

        # Put each recording in the fold with the fewest segments of this class.
        for recording in class_recordings:
            smallest = min(segment_counts)
            candidates = [i for i in range(N_FOLDS) if segment_counts[i] == smallest]
            fold_index = rng.choice(candidates)
            folds[fold_index].append(recording)
            segment_counts[fold_index] += recording["segments"]

    # Check complete coverage, no recording overlap, and all classes in each fold.
    assigned_ids = []
    for fold in folds:
        for recording in fold:
            assigned_ids.append(recording["id"])
    all_ids = {r["id"] for r in recordings}
    if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != all_ids:
        raise ValueError("Recordings are duplicated or missing from the folds.")
    for fold in folds:
        if {r["class"] for r in fold} != set(CLASSES):
            raise ValueError("A fold is missing a class.")

    # Each recording is validation once and training in the other two runs.
    runs = []
    for fold_index, fold in enumerate(folds):
        validation_ids = sorted(r["id"] for r in fold)
        runs.append({
            "fold": fold_index + 1,
            "train_recording_ids": sorted(all_ids - set(validation_ids)),
            "validation_recording_ids": validation_ids,
        })

    result = {
        "seed": seed,
        "strategy": "Three recording-grouped folds, balancing segments within each class",
        "runs": runs,
    }
    validate_assignment(runs, recording_ids, classes)
    if shuffle_recordings:
        result["strategy"] = "Shuffled within class; greedily balance segment counts"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Each cell shows: segments / recordings\n")
    print(f"{'Class':<12}" + "".join(f"{'Fold ' + str(i + 1):>20}" for i in range(N_FOLDS)))
    for class_name in CLASSES:
        cells = []
        for fold in folds:
            selected = [r for r in fold if r["class"] == class_name]
            cells.append(f"{sum(r['segments'] for r in selected):,} / {len(selected)}")
        print(f"{class_name:<12}" + "".join(f"{cell:>20}" for cell in cells))
    if output_path is not None:
        print(f"\nSaved: {output_path}")
    return result


def make_repeated_splits(data_path, original_path, output_path, seeds):
    """Keep original folds verbatim; randomize actual membership in extra runs.

    Merely changing tie-breaking seeds in largest-first allocation can produce
    the same partition with renamed folds. Shuffling recording order changes
    membership; duplicate partitions are explicitly rejected below.
    """
    if output_path.resolve() == original_path.resolve():
        raise ValueError("Repeated splits must not overwrite the original benchmark.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as data:
        validate_assignment(original["runs"], data["recording_id"], data["task_class"])
    if len(set(seeds)) != 3 or original["seed"] in seeds:
        raise ValueError("Use three distinct robustness seeds, different from development.")
    development = original.copy()
    development["assignment"] = "development"
    development["role"] = "development"
    assignments = [development]
    for seed in seeds:
        print(f"\nRobustness assignment, seed {seed}")
        assignment = make_splits(data_path, None, seed, shuffle_recordings=True)
        assignment["assignment"] = f"robustness_{seed}"
        assignment["role"] = "robustness"
        assignments.append(assignment)

    # Ignore fold numbering when checking whether assignments really differ.
    partitions = []
    for assignment in assignments:
        validation_groups = []
        for run in assignment["runs"]:
            validation_groups.append(tuple(sorted(run["validation_recording_ids"])))
        partition = tuple(sorted(validation_groups))
        if partition in partitions:
            raise ValueError("Duplicate partitions detected; use different fixed robustness seeds.")
        partitions.append(partition)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"assignments": assignments}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {output_path}; original development folds preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "radar_segments.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeated", action="store_true")
    parser.add_argument("--original", type=Path, default=ROOT / "research" / "recording_folds.json")
    parser.add_argument("--robustness-seeds", type=int, nargs=3, default=[7, 21, 84])
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / ("repeated_folds.json" if args.repeated else "recording_folds.json")
    if not output.resolve().is_relative_to(ROOT / "runs") or output.exists():
        raise ValueError("Choose a new split-output file under runs/; stored assignments are protected.")
    if args.repeated:
        make_repeated_splits(args.data, args.original, output, args.robustness_seeds)
    else:
        make_splits(args.data, output, args.seed)
