"""Verify a rectangular candidate without installing it or training any models.

Requires the supplied NPZ, completed CNN fold outputs and full-data candidate.
Uses NumPy/PyTorch only; runs inference checks and a runtime-only temporary copy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch

from radar_model.inference import average_probabilities, segment_probabilities
from radar_model.model import load_checkpoint
from radar_model.preprocessing import CLASSES, build_aligned_inputs
from verify import compare, construct_tracks, select_batch

ROOT = Path(__file__).resolve().parent


def reference_inputs(raw):
    """Independent no-taper FFT/shift calculation for numerical preprocessing parity."""
    signals = raw.astype(np.complex128)
    valid_segments = np.isfinite(signals).all(axis=(1, 2))
    signals[~valid_segments] = 0
    coefficients = np.fft.fft(signals, axis=2)
    coefficients = np.fft.fftshift(coefficients, axes=2)
    power = np.abs(coefficients)**2 / 256**2
    aligned = np.zeros_like(power)
    for row in range(len(raw)):
        shift = 0
        if power[row, 2].sum() > 0:
            shift = 128 - int(power[row, 2].argmax())
        # Slice explicitly, without circular roll or the production shift helper.
        source_start = max(0, -shift)
        source_stop = min(256, 256 - shift)
        aligned[row, :, source_start + shift:source_stop + shift] = power[row, :, source_start:source_stop]
    total = aligned.sum(axis=(1, 2), keepdims=True)
    fractions = np.divide(aligned, total, out=np.zeros_like(aligned), where=total > 0)
    return (10 * np.log10(np.maximum(fractions, 1e-8)) / 80).astype(np.float32)


def check_folds(directory, data_path):
    """Reproduce held-out probabilities with matching fold weights and preprocessing."""
    configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    if configuration["window"] != "rectangular" or configuration["backend"] != "fft":
        raise ValueError("Require rectangular FFT evaluation results.")
    assignments = json.loads((directory / "splits.json").read_text(encoding="utf-8"))["assignments"]
    with np.load(data_path, allow_pickle=False) as data:
        raw = data["segments"]
        recordings = data["recording_id"]
        times = data["time_s"]
    tracks = construct_tracks(recordings, times)
    for assignment in assignments:
        seen = []
        for run in assignment["runs"]:
            folder = directory / assignment["assignment"] / "cnn_peak_aligned" / f"fold{run['fold']}"
            model, checkpoint = load_checkpoint(folder / "model.pt")
            if model.spectral_window != "rectangular":
                raise ValueError("A fold checkpoint uses another window.")
            train_ids = set(run["train_recording_ids"])
            valid_ids = set(run["validation_recording_ids"])
            if (train_ids & valid_ids or train_ids | valid_ids != set(recordings.tolist())
                    or set(checkpoint["train_recording_ids"]) != train_ids
                    or set(checkpoint["validation_recording_ids"]) != valid_ids
                    or checkpoint["fold"] != run["fold"] or checkpoint["assignment"] != assignment["assignment"]):
                raise ValueError("Held-out checkpoint/split isolation differs.")
            expected_rows = np.flatnonzero(np.isin(recordings, sorted(valid_ids)))
            with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["classes"], CLASSES) or not np.array_equal(saved["segment_index"], expected_rows):
                    raise ValueError("Saved probability class/row order differs.")
                inputs = build_aligned_inputs(raw[expected_rows], window=model.spectral_window)
                actual = segment_probabilities(model, inputs)
                compare(actual, saved["probabilities"], f"{assignment['assignment']}/fold{run['fold']}: saved segment probabilities")
                sample = np.arange(min(64, len(inputs)))
                compare(segment_probabilities(model, inputs[sample], 1), actual[sample], "fold batch 1 vs 256")
                # Complete pieces stay in the held-out recording set.
                if set(tracks[expected_rows]) & set(tracks[np.isin(recordings, sorted(train_ids))]):
                    raise ValueError("A piece crosses training/validation recordings.")
            seen.extend(expected_rows.tolist())
        if not np.array_equal(np.sort(seen), np.arange(len(raw))):
            raise ValueError("Each segment must be held out exactly once per assignment.")


def check_runtime_batch():
    """Run API checks in the temporary directory, without research dependencies."""
    from predict import predict
    from radar_model.inference import predict_probabilities
    from radar_model.model import load_model

    with np.load("batch.npz", allow_pickle=False) as data:
        raw = data["segments"]
        tracks = data["track_id"]
        times = data["time_s"]
        expected = data["expected"]

    started = perf_counter()
    actual = predict_probabilities(raw, tracks, times)
    cold_seconds = perf_counter() - started
    assert load_model().spectral_window == "rectangular"
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert np.array_equal(actual.argmax(1), expected.argmax(1))

    ranges = np.zeros(len(raw))
    near_edge = np.zeros(len(raw), dtype=bool)
    started = perf_counter()
    labels = np.asarray(predict(raw, ranges, near_edge, tracks, times))
    warm_seconds = perf_counter() - started
    assert len(labels) == len(raw) and set(labels) <= set(CLASSES)

    order = np.random.default_rng(42).permutation(len(raw))
    shuffled = np.asarray(predict(
        raw[order], ranges[order], near_edge[order], tracks[order], times[order],
    ))
    assert np.array_equal(shuffled[np.argsort(order)], labels)
    for identifier in np.unique(tracks)[:5]:
        selected = tracks == identifier
        alone = predict_probabilities(raw[selected], tracks[selected], times[selected])
        assert np.allclose(alone, actual[selected], atol=1e-6, rtol=1e-5)
        assert np.array_equal(alone.argmax(1), actual[selected].argmax(1))

    assert predict(raw[:0], ranges[:0], near_edge[:0], tracks[:0], times[:0]) == []
    for length in (1, 30):
        piece = np.repeat(raw[:1], length, axis=0)
        piece_ranges = np.zeros(length)
        piece_edges = np.zeros(length, dtype=bool)
        piece_ids = np.zeros(length, dtype=int)
        piece_times = np.arange(length)*.1
        result = predict(piece, piece_ranges, piece_edges, piece_ids, piece_times)
        assert len(result) == length and len(set(result)) == 1
    assert not Path("research").exists()

    print(json.dumps({
        "status": "PASS",
        "window": "rectangular",
        "n_segments": len(raw),
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "device": "cpu",
        "runtime_only_copy": True,
        "ordering_and_labels": True,
    }, indent=2))


def verify_candidate(args):
    """Check full-data provenance, parity, ordering and CPU inference in isolation."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checkpoint_path = args.model_dir / "model.pt"
    config = json.loads((args.model_dir / "model_config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if (config.get("window") != "rectangular" or config["checkpoint_sha256"] != digest
            or config["classes"] != list(CLASSES)
            or config["model"] != "peak_aligned_cnn_whole_piece_probability_averaging"):
        raise ValueError("Candidate configuration/hash differs.")
    model, checkpoint = load_checkpoint(checkpoint_path)
    training = checkpoint.get("training", {})
    if (model.spectral_window != "rectangular" or "fold" in checkpoint
            or training.get("n_segments") != 61673 or training.get("n_recordings") != 87
            or training.get("seed") != 42 or training.get("epochs") != 20
            or training.get("batch_size") != 256 or training.get("optimizer") != "Adam"
            or training.get("learning_rate") != .001 or training.get("weight_decay") != .0001
            or training.get("early_stopping") is not False or training.get("validation_split") is not False):
        raise ValueError("Candidate is not the declared fixed full-data fit.")
    check_folds(args.fold_results, args.data)
    _, (raw, tracks, times) = select_batch(args.data, args.n)
    # Include non-finite/zero cases and extreme peak bins without scoring accuracy.
    chirps = np.arange(256)
    tone_signals = []
    for bin_number in (-128, -127, 0, 127):
        tone = np.exp(2j * np.pi * bin_number * chirps / 256)
        tone_signals.append(np.tile(tone, (5, 1)))
    tones = np.stack(tone_signals).astype(np.complex64)
    special = np.zeros((2, 5, 256), dtype=np.complex64)
    special[1, 0, 0] = np.nan
    sample = np.concatenate([raw[:64], tones, special])
    if not np.array_equal(build_aligned_inputs(sample, window="rectangular"), reference_inputs(sample)):
        raise AssertionError("Rectangular preprocessing differs from independent FFT reference.")
    inputs = build_aligned_inputs(raw, window=model.spectral_window)
    probabilities = segment_probabilities(model, inputs)
    expected = average_probabilities(probabilities, tracks, times)
    # A fresh process gets inference code, model assets and verification helpers,
    # but no research directory. It cannot accidentally import research helpers.
    with tempfile.TemporaryDirectory(prefix="radar_rectangular_verification_") as temporary:
        runtime = Path(temporary)
        shutil.copy2(ROOT / "predict.py", runtime / "predict.py")
        shutil.copytree(ROOT / "radar_model", runtime / "radar_model", ignore=shutil.ignore_patterns("__pycache__"))
        (runtime / "model").mkdir()
        for name in ("model.pt", "model_config.json"):
            shutil.copy2(args.model_dir / name, runtime / "model" / name)
        np.savez(runtime / "batch.npz", segments=raw, track_id=tracks, time_s=times, expected=expected)
        for name in ("verify.py", "verify_rectangular_candidate.py"):
            shutil.copy2(ROOT / name, runtime / name)
        # -I prevents user paths/environment variables from supplying imports.
        launcher = (
            "import sys; sys.path.insert(0, '.'); "
            "from verify_rectangular_candidate import check_runtime_batch; "
            "check_runtime_batch()"
        )
        subprocess.run([sys.executable, "-I", "-c", launcher], cwd=runtime, check=True)
    print(f"Verified candidate SHA-256: {digest}. No installed model or report was changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "runs/final_rectangular_candidate")
    parser.add_argument("--fold-results", type=Path, default=ROOT / "runs/cnn_rectangular")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n", type=int, default=15000)
    verify_candidate(parser.parse_args())
