"""Assignment entry point: one string label per original input segment."""

from radar_model.inference import predict as _predict


def predict(segments, range_m, near_edge, track_id, time_s):
    """Classify complete supplied pieces on CPU, preserving input row order."""
    return _predict(segments, range_m, near_edge, track_id, time_s)
