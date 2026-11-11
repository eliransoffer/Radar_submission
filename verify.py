"""CPU timing and file checks; no fitting or accuracy claims."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from radar_model import inference
from radar_model.model import load_model
from radar_model.preprocessing import CLASSES, build_aligned_inputs

ROOT = Path(__file__).resolve().parent


def construct_tracks(recordings, times):
    """Reconstruct assignment-style pieces for verification data only; labels are unused."""
    tracks = np.empty(len(times), dtype=np.int64)
    next_track = 0
    for recording in np.unique(recordings):
        rows = np.flatnonzero(recordings == recording)
        rows = rows[np.argsort(times[rows], kind="stable")]
        length = 0
        previous = None
        for row in rows:
            gap = previous is not None and times[row] - previous > .5
            if length == 0 or length == 30 or gap:
                identifier = next_track
                next_track += 1
                length = 0
            tracks[row] = identifier
            length += 1
            previous = times[row]
    return tracks


def select_batch(path, count, allowed_recordings=None):
    """Load real complete pieces before timing, covering recordings before filling randomly."""
    with np.load(path, allow_pickle=False) as data:
        recordings = data["recording_id"]
        times = data["time_s"]
        tracks = construct_tracks(recordings, times)
        allowed = np.ones(len(times), dtype=bool)
        if allowed_recordings is not None:
            allowed = np.isin(recordings, allowed_recordings)
        available = np.unique(tracks[allowed])
        first_pieces = []
        for recording in np.unique(recordings[allowed]):
            rows = np.flatnonzero((recordings == recording) & allowed)
            first_pieces.append(int(tracks[rows[0]]))
        remaining = np.setdiff1d(available, first_pieces)
        order = first_pieces + np.random.default_rng(42).permutation(remaining).tolist()
        chosen = []
        length = 0
        for identifier in order:
            rows = np.flatnonzero(tracks == identifier)
            chosen.extend(rows.tolist())
            length += len(rows)
            if length >= count:
                break
        selected = np.random.default_rng(21).permutation(np.array(chosen, dtype=np.int64))
        arrays = (data["segments"][selected], tracks[selected], times[selected])
    return selected, arrays


def compare(actual, expected, description):
    """Require float probability parity and identical class decisions, allowing CPU roundoff."""
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise AssertionError(f"{description}: numerical mismatch.")
    if not np.array_equal(actual.argmax(1), expected.argmax(1)):
        raise AssertionError(f"{description}: changed class decisions.")
    print(json.dumps({"check": description, "max_absolute_error": float(np.abs(actual - expected).max()), "status": "PASS"}))


def benchmark(args):
    """Time CPU loading, preprocessing, CNN, aggregation and warm API calls separately."""
    _, (raw, tracks, times) = select_batch(args.data, args.n)
    # Compressed dataset loading/decompression and piece selection are already finished.
    load_model.cache_clear()
    started = perf_counter()
    model = load_model()
    loaded = perf_counter()
    inputs = build_aligned_inputs(raw, window=model.spectral_window)
    transformed = perf_counter()
    segment = inference.segment_probabilities(model, inputs)
    inferred = perf_counter()
    averaged = inference.average_probabilities(segment, tracks, times)
    labels = np.asarray(CLASSES)[averaged.argmax(1)].tolist()
    finished = perf_counter()
    warm_started = perf_counter()
    warm = inference.predict_probabilities(raw, tracks, times)
    warm_seconds = perf_counter() - warm_started
    compare(warm, averaged, "warm API vs timed stage pipeline")
    if len(labels) != len(raw):
        raise AssertionError("Incorrect label count.")
    print(json.dumps({"n_segments": len(raw), "n_pieces": len(np.unique(tracks)), "cpu_threads": torch.get_num_threads(),
                      "device": "cpu", "processor": platform.processor(), "platform": platform.platform(),
                      "numpy": str(np.__version__), "torch": str(torch.__version__),
                      "model_loading_seconds": loaded - started, "fft_alignment_seconds": transformed - loaded,
                      "cnn_seconds": inferred - transformed, "aggregation_and_labels_seconds": finished - inferred,
                      "cold_stage_total_seconds": finished - started, "warm_api_seconds": warm_seconds,
                      "dataset_loading_included": False}, indent=2))


def contents(args):
    """List package files and require the model/report/configuration before hand-in."""
    required = ["predict.py", "requirements.txt", "README.md", ".gitignore", "report.pdf", "model/model.pt", "model/model_config.json"]
    missing = [name for name in required if not (ROOT / name).is_file()]
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not any(part in (".venv", "__pycache__") for part in path.relative_to(ROOT).parts):
            print(f"{path.relative_to(ROOT).as_posix()}  {path.stat().st_size} bytes")
    if missing:
        raise FileNotFoundError("Incomplete submission: " + ", ".join(missing))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("benchmark", "contents"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--n", type=int, default=15000)
    args = parser.parse_args()
    if args.action == "benchmark" and args.data is None:
        parser.error("--data is required")
    if args.action == "benchmark":
        benchmark(args)
    else:
        contents(args)
