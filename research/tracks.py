"""Validated recording-to-piece construction and local/whole probability means.

Research helper only. Production inference has no import from this module.
"""
import numpy as np
MAX_TRACK_LENGTH = 30
MAX_GAP_S = 0.5
LOCAL_WINDOW = 5

def construct_tracks(recording_id, time_s):
    """Simulate track IDs, returning one ID per original input row.

    Recordings are sorted separately by time, with original row index breaking
    equal-time ties. A gap strictly greater than 0.5 s resets the 30-row counter.
    No labels or probabilities influence construction. Recording IDs are only
    needed here to simulate the track_id supplied by the prediction API.
    """
    recording_id = np.asarray(recording_id)
    time_s = np.asarray(time_s)
    tracks = np.empty(len(time_s), dtype=np.int64)
    next_track = 0
    for recording in np.unique(recording_id):
        indices = np.flatnonzero(recording_id == recording)
        indices = indices[np.argsort(time_s[indices], kind="stable")]
        length = 0
        previous_time = None
        for index in indices:
            gap = previous_time is not None and time_s[index] - previous_time > MAX_GAP_S
            if length == 0 or length == MAX_TRACK_LENGTH or gap:
                track = next_track
                next_track += 1
                length = 0
            tracks[index] = track
            length += 1
            previous_time = time_s[index]
    return tracks

def average_track_probabilities(probabilities, track_id, time_s, method="whole_track"):
    """Return (N,9) probabilities in input order, using only these API inputs.

    method='whole_track': mean over the full supplied track piece.
    method='local_five': symmetric +/-2 observations, clipped at track edges.
    Collect all probabilities for a track before calling; splitting one track
    across separate calls loses its context. Model inference batches can have
    any ordering if probabilities are restored to their corresponding rows.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    track_id = np.asarray(track_id)
    time_s = np.asarray(time_s)
    if probabilities.shape != (len(track_id), 9) or time_s.shape != track_id.shape:
        raise ValueError("Expected probabilities (N,9), track_id (N,), time_s (N,).")
    if method not in ("whole_track", "local_five"):
        raise ValueError("method must be whole_track or local_five.")
    averaged = np.empty_like(probabilities)
    for track in np.unique(track_id):
        indices = np.flatnonzero(track_id == track)
        # Equal-time rows have no temporal order. Break ties by probability
        # vector to keep local output deterministic even if input rows shuffle.
        sort_keys = []
        for column in range(8, -1, -1):
            sort_keys.append(probabilities[indices, column])
        sort_keys.append(time_s[indices])
        order = np.lexsort(tuple(sort_keys))
        indices = indices[order]
        if len(indices) > MAX_TRACK_LENGTH or np.any(np.diff(time_s[indices]) > MAX_GAP_S):
            raise ValueError("A supplied track exceeds 30 segments or spans a gap >0.5 s.")
        if method == "whole_track":
            averaged[indices] = probabilities[indices].mean(axis=0)
        else:
            radius = LOCAL_WINDOW // 2
            for position, index in enumerate(indices):
                neighbors = indices[max(0, position - radius):position + radius + 1]
                averaged[index] = probabilities[neighbors].mean(axis=0)
    return averaged
