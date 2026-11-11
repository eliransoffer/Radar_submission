"""Fixed CNN training helpers reused from cnn_baseline.py; no parameter search."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from radar_model.model import make_cnn
from radar_model.preprocessing import CLASSES, build_aligned_inputs
from research.spectral import doppler_spectra
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

def build_cnn_inputs(segments, window="rectangular"):
    """Return float32 (N, 5, 256) log Doppler power, preserving input order.

    Apply the shared selected-window slow-time FFT to every range cell. Divide
    by TOTAL spectral power across all five cells: this removes overall echo
    strength while preserving relative power across range cells. Clip at -80 dB
    and divide by 80, yielding [-1, 0]. These are fixed per-segment operations,
    so no training/validation statistics are involved. Keep DC and peak position.
    Zero/non-finite segments receive the fixed floor in every bin.
    """
    result = np.empty((len(segments), 5, 256), dtype=np.float32)
    for start in range(0, len(segments), 1024):
        stop = min(start + 1024, len(segments))
        signals = segments[start:stop].astype(np.complex128)
        valid = np.isfinite(signals).all(axis=(1, 2))
        signals[~valid] = 0
        _, power, _ = doppler_spectra(signals.reshape(-1, 256), window=window)
        power = power.reshape(-1, 5, 256)
        total = power.sum(axis=(1, 2), keepdims=True)
        fractions = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
        log_power = 10 * np.log10(np.maximum(fractions, 1e-8))
        result[start:stop] = (log_power / 80).astype(np.float32)
    return result

def make_loader(inputs, targets, indices, shuffle, seed):
    """Reuse the evaluated seeded CPU loader, retaining exact sample and shuffle order."""
    # Use the evaluated loader and seeded shuffle order, without worker processes.
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()), batch_size=BATCH_SIZE,
        shuffle=shuffle, generator=generator, num_workers=0,
    )


def fit_fixed(inputs, targets):
    """Run the evaluated fit_network fixed-budget branch: fresh seed-42 CNN, 20 epochs."""
    torch.manual_seed(42)
    model = make_cnn().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    training_loader = make_loader(inputs, targets, np.arange(len(targets)), shuffle=True, seed=42)
    history = []
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in training_loader:
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_targets)
        history.append({"epoch": epoch, "train_loss": total_loss / len(targets)})
        if epoch % 5 == 0:
            print(f"    epoch {epoch}: training loss {history[-1]['train_loss']:.4f}", flush=True)
    return model.cpu().eval(), history


def checkpoint_configuration(model, window="rectangular"):
    """Return the recorded aligned-CNN format and fixed settings, without fitted metadata."""
    return {
        "experiment": "peak_aligned_doppler_cnn",
        "classes": list(CLASSES),
        "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        "architecture": str(model),
        "window": window,
        "model_seed": 42,
        "epochs": 20,
        "batch_size": 256,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "alignment": "central-cell dominant bin shifted to zero; same integer shift for all cells",
        "padding": "zero linear power; out-of-band power discarded; fixed 256 bins",
        "normalization": "retained five-cell total power, then -80 dB floor and /80",
        "peak_frequency_is_classifier_input": False,
    }


def train_final(args):
    """Optionally reproduce the full-data fit in runs/; never replace submitted weights."""
    from research.feature_extraction import new_output
    output = new_output(args.output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with np.load(args.data, allow_pickle=False) as data:
        labels, recordings = data["task_class"], data["recording_id"]
        if len(labels) != 61673 or len(np.unique(recordings)) != 87 or set(labels) != set(CLASSES):
            raise ValueError("Expected the complete supplied 61,673-segment, 87-recording subset.")
        targets = np.array([CLASSES.index(str(name)) for name in labels], dtype=np.int64)
        inputs = build_aligned_inputs(data["segments"], window=args.window)
    model, history = fit_fixed(inputs, targets)
    training = {"n_segments": len(targets), "n_recordings": len(np.unique(recordings)),
                "recording_ids": np.unique(recordings).tolist(), "seed": 42, "epochs": 20,
                "batch_size": 256, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "early_stopping": False, "validation_split": False,
                "cpu_threads": 4, "deterministic_algorithms": True}
    checkpoint = output / "model.pt"
    saved_model = checkpoint_configuration(model, args.window)
    saved_model["training"] = training
    saved_model["state_dict"] = model.state_dict()
    torch.save(saved_model, checkpoint)
    configuration = {"model": "peak_aligned_cnn_whole_piece_probability_averaging", "classes": list(CLASSES),
                     "window": args.window, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output / "model_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    print(f"Saved an optional reproduction candidate at {output}; submitted model/ is untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Complete supplied radar_segments.npz.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/final_candidate")
    parser.add_argument("--window", choices=("rectangular",), default="rectangular")
    train_final(parser.parse_args())
