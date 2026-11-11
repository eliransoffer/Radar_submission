"""Optional fixed-model reproduction on the saved recording-grouped assignments.

Run classical or cnn as documented in research/README.md; supplied subset only.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from radar_model.inference import segment_probabilities
from radar_model.model import make_cnn
from radar_model.preprocessing import CLASSES, build_aligned_inputs
from research.classical import evaluate_predictions, make_models
from research.cnn import build_cnn_inputs, checkpoint_configuration, fit_fixed
from research.feature_extraction import FEATURE_NAMES, new_output
from research.make_splits import validate_assignment
from research.tracks import average_track_probabilities, construct_tracks

REPRESENTATIONS = {"original": 5, "central": 9, "spatial": 17, "G1": 25, "G2": 57, "G3": 65}
METRICS = ["macro_f1", "accuracy", "recording_balanced_accuracy"]


def assignments(path, metadata, scope):
    """Use the stored development/robustness allocations; validate every recording boundary."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if "assignments" in document:
        items = document["assignments"]
    else:
        development = document.copy()
        development["assignment"] = "development"
        development["role"] = "development"
        items = [development]
    selected = []
    for item in items:
        if scope != "all" and item["role"] != scope:
            continue
        folds = validate_assignment(item["runs"], metadata.recording_id.to_numpy(), metadata.true_class.to_numpy())
        selected.append((item, folds))
    if not selected:
        raise ValueError("No assignments match the requested scope.")
    return selected


def record_scores(targets, predicted, metadata, identity, split, folder):
    """Reuse original class/recording/confusion metrics with equal-class recording accuracy."""
    folder.mkdir(parents=True, exist_ok=True)
    metrics, classes, records = evaluate_predictions(targets, predicted, metadata.recording_id.to_numpy(),
                                                    identity["model"], identity["fold"], split, folder)
    # Equal weight for each recording within a class, then for each class.
    record_table = pd.DataFrame(records)
    class_accuracy = record_table.groupby("class").accuracy.mean()
    metrics["recording_balanced_accuracy"] = class_accuracy.mean()
    metrics.update(identity)
    for row in classes + records:
        row.update(identity)
    return metrics, classes, records


def prediction_rows(metadata, predicted, identity):
    """Retain original indices and fold provenance for held-out diagnostics, not model input."""
    rows = metadata.copy()
    rows["predicted_class"] = np.asarray(CLASSES)[predicted]
    for name, value in identity.items():
        rows[name] = value
    return rows


def save_results(metrics, classes, records, predictions, output):
    """Separate mean-fold, pooled OOF and assignment-mean summaries without new tuning."""
    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(records).to_csv(output / "per_recording_accuracy.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    keys = ["role", "assignment", "seed", "window", "model", "representation", "method"]
    # First summarize folds within each assignment; keep train/validation apart.
    summary = metrics.groupby(keys + ["split"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{name}_{statistic}" for name, statistic in summary.columns]
    summary.to_csv(output / "mean_fold_metrics.csv")
    pooled = []
    # Pooled OOF scores use concatenated predictions, not averages of fold scores.
    for identity, group in predictions.groupby(keys):
        if group.segment_index.duplicated().any():
            raise ValueError("An observation received multiple validation predictions in one assignment.")
        correct = group.true_class == group.predicted_class
        recording_accuracy = group.assign(correct=correct).groupby(["true_class", "recording_id"]).correct.mean()
        row = dict(zip(keys, identity))
        row["n_segments"] = len(group)
        row["macro_f1"] = f1_score(
            group.true_class, group.predicted_class,
            labels=CLASSES, average="macro", zero_division=0,
        )
        row["accuracy"] = accuracy_score(group.true_class, group.predicted_class)
        row["recording_balanced_accuracy"] = recording_accuracy.groupby(level=0).mean().mean()
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.csv", index=False)
    means = metrics[metrics.split == "validation"].groupby(keys)[METRICS].mean().reset_index()
    # Reused recordings make this SD descriptive, not an independent-test interval.
    across = means.groupby(["role", "model", "representation", "method"])[METRICS].agg(["mean", "std"])
    across.columns = [f"{name}_{statistic}" for name, statistic in across.columns]
    across.to_csv(output / "assignment_mean_summary.csv")


def classical(args):
    """Fit fixed LR/XGBoost folds and evaluate three decision-level contexts.

    Training scores use independent segments. Held-out scores also include
    local-five and whole-piece probability means, using the same track pieces.
    """
    configuration = json.loads((args.features / "configuration.json").read_text(encoding="utf-8"))
    if configuration["feature_names"] != list(FEATURE_NAMES) or configuration["backend"] != "fft":
        raise ValueError("Feature order/backend differs from the reported definitions.")
    if configuration.get("window") != args.window:
        raise ValueError("Feature cache window differs from the requested evaluation.")
    metadata = pd.read_csv(args.features / "segments.csv")
    with np.load(args.features / "features.npz", allow_pickle=False) as cache:
        matrix = cache["features"]
        if not np.array_equal(cache["segment_index"], metadata.segment_index) or matrix.shape != (len(metadata), 65):
            raise ValueError("Feature cache row order/shape differs from metadata.")
    selected = assignments(args.folds, metadata, args.scope)
    output = new_output(args.output)
    (output / "configuration.json").write_text(json.dumps({"window": args.window, "backend": "fft",
        "representations": REPRESENTATIONS, "feature_cache": str(args.features.resolve())}, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps({"assignments": [item for item, _ in selected]}, indent=2), encoding="utf-8")
    targets = np.array([CLASSES.index(name) for name in metadata.true_class], dtype=np.int64)
    metrics, class_rows, recordings, predictions = [], [], [], []
    for assignment, folds in selected:
        for representation, size in REPRESENTATIONS.items():
            features = matrix[:, :size]
            for fold, train, validation in folds:
                for name, pipeline in make_models().items():
                    identity = {
                        "assignment": assignment["assignment"],
                        "role": assignment["role"],
                        "seed": assignment["seed"],
                    }
                    identity.update(window=args.window, model=name, representation=representation, method="segment", fold=fold)
                    folder = output / assignment["assignment"] / representation / name / f"fold{fold}"
                    print(f"{assignment['assignment']}/{representation}/{name}/fold{fold}", flush=True)
                    # Fitted imputation and LR standardization see training recordings only.
                    pipeline.fit(features[train], targets[train])
                    if not np.array_equal(pipeline.classes_, np.arange(9)):
                        raise ValueError("Fitted classifier class order differs.")
                    for split, indices in (("train", train), ("validation", validation)):
                        independent_labels = pipeline.predict(features[indices]).astype(np.int64)
                        if split == "validation":
                            independent = pipeline.predict_proba(features[indices]).astype(np.float64)
                            if not np.array_equal(independent.argmax(axis=1), independent_labels):
                                raise ValueError("Class predictions differ from ordered probability argmax.")
                            folder.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(folder / "validation_probabilities.npz", segment_index=indices,
                                                probabilities=independent, classes=np.asarray(CLASSES))
                        methods = ("segment",)
                        if split == "validation":
                            methods = ("segment", "local_five", "whole_track")
                        for method in methods:
                            predicted = independent_labels
                            if method != "segment":
                                averaged = average_track_probabilities(
                                    independent, metadata.track_id.to_numpy()[indices],
                                    metadata.time_s.to_numpy()[indices], method,
                                )
                                predicted = averaged.argmax(axis=1)
                            current = identity.copy()
                            current["method"] = method
                            values, per_class, per_recording = record_scores(targets[indices], predicted,
                                metadata.iloc[indices], current, split, folder / method)
                            metrics.append(values)
                            class_rows.extend(per_class)
                            recordings.extend(per_recording)
                            if split == "validation":
                                predictions.append(prediction_rows(metadata.iloc[indices], predicted, current))
                    checkpoint = {
                        "pipeline": pipeline,
                        "classes": list(CLASSES),
                        "feature_names": list(FEATURE_NAMES[:size]),
                    }
                    checkpoint.update(identity)
                    checkpoint["train_recording_ids"] = np.unique(metadata.recording_id.to_numpy()[train]).tolist()
                    checkpoint["validation_recording_ids"] = np.unique(metadata.recording_id.to_numpy()[validation]).tolist()
                    joblib.dump(checkpoint, folder / "pipeline.joblib")
    save_results(metrics, class_rows, recordings, predictions, output)


def reuse_aligned_fold(paths, assignment, fold, window, train_ids, validation_ids):
    """Reuse exactly one matching ablation fold, checking settings and held-out IDs."""
    candidates = []
    for path in paths:
        checkpoint_path = path / assignment["assignment"] / window / "cnn_peak_aligned" / f"fold{fold}.pt"
        if checkpoint_path.is_file():
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one saved aligned CNN for this assignment/window/fold.")
    source = candidates[0]
    configuration = json.loads((source / "configuration.json").read_text(encoding="utf-8"))
    training = configuration["neural_training"]
    if (configuration["models"] != "cnn" or configuration["transform_method"] != "fft"
            or configuration["model_seed"] != 42 or configuration["neural_epochs"] != 20
            or training["learning_rate"] != .001 or training["weight_decay"] != .0001
            or training["cnn_batch_size"] != 256 or window not in configuration["windows"]
            or assignment not in configuration["assignments"]):
        raise ValueError("Saved aligned CNN does not use the selected fixed protocol/splits.")
    folder = source / assignment["assignment"] / window / "cnn_peak_aligned"
    checkpoint = torch.load(folder / f"fold{fold}.pt", map_location="cpu", weights_only=True)
    expected = {"assignment": assignment["assignment"], "seed": assignment["seed"], "role": assignment["role"],
                "fold": fold, "window": window, "model": "cnn_peak_aligned", "classes": list(CLASSES),
                "transform_method": "fft", "input_representation": "aligned five-cell log power"}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("Saved checkpoint provenance differs.")
    if set(checkpoint["train_recording_ids"]) != set(train_ids) or set(checkpoint["validation_recording_ids"]) != set(validation_ids):
        raise ValueError("Saved checkpoint recording separation differs.")
    model = make_cnn().cpu()
    if checkpoint["architecture"] != str(model):
        raise ValueError("Saved CNN architecture differs.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    history = pd.read_csv(folder / f"fold{fold}_history.csv")
    if not np.array_equal(history.epoch, np.arange(1, 21)):
        raise ValueError("Saved history is not the fixed 20-epoch fit.")
    return model.eval(), history.to_dict("records"), str((folder / f"fold{fold}.pt").resolve())


def cnn(args):
    """Reproduce fresh original/aligned CNN folds, then fixed probability averaging."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with np.load(args.data, allow_pickle=False) as data:
        raw = data["segments"]
        metadata = pd.DataFrame({"segment_index": np.arange(len(raw)), "recording_id": data["recording_id"],
                                 "true_class": data["task_class"].astype(str), "time_s": data["time_s"]})
    metadata["track_id"] = construct_tracks(metadata.recording_id.to_numpy(), metadata.time_s.to_numpy())
    targets = np.array([CLASSES.index(name) for name in metadata.true_class], dtype=np.int64)
    selected = assignments(args.folds, metadata, args.scope)
    output = new_output(args.output)
    (output / "configuration.json").write_text(json.dumps({"window": args.window, "backend": "fft",
        "cnn_variants": args.cnn_variants, "saved_cnn_runs": [str(path.resolve()) for path in args.saved_cnn_runs]}, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps({"assignments": [item for item, _ in selected]}, indent=2), encoding="utf-8")
    metrics, class_rows, recordings, predictions = [], [], [], []
    for name, transform in (("cnn_original", build_cnn_inputs), ("cnn_peak_aligned", build_aligned_inputs)):
        if args.cnn_variants != "both" and name != f"cnn_{args.cnn_variants}":
            continue
        inputs = transform(raw, window=args.window)
        for assignment, folds in selected:
            for fold, train, validation in folds:
                source_checkpoint = None
                if name == "cnn_peak_aligned" and args.saved_cnn_runs:
                    print(f"{assignment['assignment']}/{name}/fold{fold}: reuse matching ablation checkpoint", flush=True)
                    model, history, source_checkpoint = reuse_aligned_fold(args.saved_cnn_runs, assignment, fold,
                        args.window, np.unique(metadata.recording_id.to_numpy()[train]),
                        np.unique(metadata.recording_id.to_numpy()[validation]))
                else:
                    print(f"{assignment['assignment']}/{name}/fold{fold}: fresh 20-epoch fit", flush=True)
                    model, history = fit_fixed(inputs[train], targets[train])
                folder = output / assignment["assignment"] / name / f"fold{fold}"
                folder.mkdir(parents=True)
                pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
                configuration = checkpoint_configuration(model, args.window)
                if name == "cnn_original":
                    configuration.update(experiment="original_doppler_cnn", alignment="none",
                                         padding="none; unaligned frequency bins",
                                         normalization="total five-cell power, then -80 dB floor and /80")
                checkpoint = configuration.copy()
                checkpoint["assignment"] = assignment["assignment"]
                checkpoint["role"] = assignment["role"]
                checkpoint["seed"] = assignment["seed"]
                checkpoint["fold"] = fold
                checkpoint["source_checkpoint"] = source_checkpoint
                checkpoint["state_dict"] = model.state_dict()
                checkpoint["train_recording_ids"] = np.unique(metadata.recording_id.to_numpy()[train]).tolist()
                checkpoint["validation_recording_ids"] = np.unique(metadata.recording_id.to_numpy()[validation]).tolist()
                torch.save(checkpoint, folder / "model.pt")
                for split, indices in (("train", train), ("validation", validation)):
                    independent = segment_probabilities(model, inputs[indices])
                    if split == "validation":
                        np.savez_compressed(folder / "validation_probabilities.npz", segment_index=indices,
                                            probabilities=independent, classes=np.asarray(CLASSES))
                    methods = ("segment",)
                    if split == "validation":
                        methods = ("segment", "local_five", "whole_track")
                    for method in methods:
                        probabilities = independent
                        if method != "segment":
                            probabilities = average_track_probabilities(
                                independent, metadata.track_id.to_numpy()[indices],
                                metadata.time_s.to_numpy()[indices], method,
                            )
                        predicted = probabilities.argmax(axis=1)
                        identity = {
                            "assignment": assignment["assignment"],
                            "role": assignment["role"],
                            "seed": assignment["seed"],
                        }
                        identity.update(window=args.window, model=name, representation="five_cell_spectra", method=method, fold=fold)
                        values, per_class, per_recording = record_scores(targets[indices], predicted, metadata.iloc[indices], identity, split, folder / method)
                        metrics.append(values)
                        class_rows.extend(per_class)
                        recordings.extend(per_recording)
                        if split == "validation":
                            if source_checkpoint and method == "segment":
                                # Require the new shared transform to reproduce saved ablation labels.
                                source = next(path for path in args.saved_cnn_runs if path.resolve() in Path(source_checkpoint).parents)
                                reference = pd.read_csv(source / "validation_predictions.csv")
                                reference = reference[(reference.assignment == assignment["assignment"]) &
                                    (reference.window == args.window) & (reference.fold == fold) &
                                    (reference.model == name)].sort_values("segment_index")
                                if (not np.array_equal(reference.segment_index, indices) or not
                                        np.array_equal(reference.predicted_class, np.asarray(CLASSES)[predicted])):
                                    raise ValueError("Reused aligned checkpoint does not reproduce saved held-out labels.")
                            predictions.append(prediction_rows(metadata.iloc[indices], predicted, identity))
                del model
        del inputs
    save_results(metrics, class_rows, recordings, predictions, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("classical", "cnn"))
    parser.add_argument("--data", type=Path, help="Supplied radar_segments.npz; required for CNN reproduction.")
    parser.add_argument("--features", type=Path, default=ROOT / "runs/features")
    parser.add_argument("--folds", type=Path, default=ROOT / "research/repeated_folds.json")
    parser.add_argument("--scope", choices=("development", "robustness", "all"), default="development")
    parser.add_argument("--window", choices=("rectangular",), default="rectangular")
    parser.add_argument("--cnn-variants", choices=("original", "peak_aligned", "both"), default="both")
    parser.add_argument("--saved-cnn-runs", type=Path, nargs="*", default=[],
                        help="Completed ablation directories; reuse aligned folds only, without fallback refitting.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output under runs/.")
    args = parser.parse_args()
    if args.experiment == "cnn":
        if args.data is None:
            parser.error("--data is required for CNN reproduction")
        cnn(args)
    else:
        classical(args)
