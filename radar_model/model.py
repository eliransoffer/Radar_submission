"""The evaluated two-convolution CNN; CPU checkpoint loading and caching."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .preprocessing import CLASSES

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "model/model.pt"
CONFIG_PATH = ROOT / "model/model_config.json"
CPU_THREADS = 4
BATCH_SIZE = 256


def make_cnn():
    """Five cell channels, frequency convolutions, and nine class logits."""
    return nn.Sequential(
        nn.Conv1d(5, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(16),
        nn.Flatten(),
        nn.Linear(32 * 16, len(CLASSES)),
    )


def load_checkpoint(path):
    """Load the audited aligned-CNN format; this also supports fold parity checks."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "experiment": "peak_aligned_doppler_cnn", "classes": list(CLASSES),
        "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        "peak_frequency_is_classifier_input": False,
        "alignment": "central-cell dominant bin shifted to zero; same integer shift for all cells",
        "padding": "zero linear power; out-of-band power discarded; fixed 256 bins",
        "normalization": "retained five-cell total power, then -80 dB floor and /80",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("Checkpoint is not the evaluated peak-aligned CNN.")
    model = make_cnn().cpu()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint["architecture"] != str(model):
        raise ValueError("Checkpoint architecture differs from the selected CNN.")
    for weights in model.state_dict().values():
        if not torch.isfinite(weights).all().item():
            raise ValueError("Checkpoint has non-finite weights.")
    model.spectral_window = checkpoint.get("window")
    if model.spectral_window != "rectangular":
        raise ValueError("Checkpoint must explicitly use rectangular preprocessing.")
    return model.eval(), checkpoint


@lru_cache(maxsize=1)
def load_model():
    """Load the included, hash-identified full-data model once; never download weights."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("Missing model/model.pt: the final full-data peak-aligned CNN has not been supplied.")
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    expected_configuration = {
        "model": "peak_aligned_cnn_whole_piece_probability_averaging",
        "checkpoint_sha256": digest,
        "classes": list(CLASSES),
    }
    for key, value in expected_configuration.items():
        if configuration[key] != value:
            raise ValueError("Checkpoint provenance is incomplete or differs from model_config.json.")
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    if configuration.get("window") != model.spectral_window:
        raise ValueError("Configuration and checkpoint windows differ.")
    training = checkpoint.get("training", {})
    if "fold" in checkpoint or training.get("n_segments") != 61673 or training.get("n_recordings") != 87:
        raise ValueError("Submission requires the full-data aligned CNN, not a held-out fold model.")
    return model
