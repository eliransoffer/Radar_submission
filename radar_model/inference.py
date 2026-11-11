"""CPU segment inference followed by the evaluated probability-averaging rule."""

import numpy as np
import torch

from .model import BATCH_SIZE, load_model
from .preprocessing import CLASSES, build_aligned_inputs


def segment_probabilities(model, inputs, batch_size=BATCH_SIZE):
    """Return float64 softmax probabilities in the original segment order."""
    probabilities = np.empty((len(inputs), len(CLASSES)), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size])
            logits = model(batch)
            batch_probabilities = torch.softmax(logits.double(), dim=1)
            probabilities[start:start + len(logits)] = batch_probabilities.numpy()
    return probabilities


def average_probabilities(probabilities, track_id, time_s):
    """Average nine probabilities within each supplied piece and restore original rows."""
    averaged = np.empty_like(probabilities)
    for identifier in np.unique(track_id):
        indices = np.flatnonzero(track_id == identifier)
        # Match the evaluated tie rule; whole-piece means need no causal ordering.
        sort_keys = []
        for column in range(8, -1, -1):
            sort_keys.append(probabilities[indices, column])
        # lexsort uses the last key first: time, then probability columns 0..8.
        sort_keys.append(time_s[indices])
        order = np.lexsort(tuple(sort_keys))
        indices = indices[order]
        if len(indices) > 30 or np.any(np.diff(time_s[indices]) > .5):
            raise ValueError("Each supplied piece must have <=30 segments and no gap >0.5 s.")
        # Repeated observations supply context without learned temporal parameters.
        averaged[indices] = probabilities[indices].mean(axis=0)
    return averaged


def predict_probabilities(segments, track_id, time_s):
    """Preprocess, infer and average complete pieces; return (N,9) in input order."""
    segments = np.asarray(segments)
    track_id = np.asarray(track_id)
    time_s = np.asarray(time_s)
    if segments.ndim != 3 or segments.shape[1:] != (5, 256) or not np.iscomplexobj(segments):
        raise ValueError("Expected complex segments with shape (N,5,256).")
    if track_id.shape != (len(segments),) or time_s.shape != track_id.shape:
        raise ValueError("Expected one track ID and timestamp per segment.")
    if not np.issubdtype(track_id.dtype, np.integer) or not np.isfinite(time_s).all():
        raise ValueError("Track IDs must be integers and timestamps finite.")
    if not len(segments):
        return np.empty((0, len(CLASSES)), dtype=np.float64)
    model = load_model()
    inputs = build_aligned_inputs(segments, window=model.spectral_window)
    probabilities = segment_probabilities(model, inputs)
    return average_probabilities(probabilities, track_id, time_s)


def predict(segments, range_m, near_edge, track_id, time_s):
    """Return N class strings; range and near_edge are accepted but not predictive."""
    if np.shape(range_m) != (len(segments),) or np.shape(near_edge) != (len(segments),):
        raise ValueError("Expected one range and near-edge flag per segment.")
    probabilities = predict_probabilities(segments, track_id, time_s)
    class_indices = probabilities.argmax(axis=1)
    labels = np.asarray(CLASSES)[class_indices]
    return labels.tolist()
