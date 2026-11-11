# Radar target classification

Nine-class classifier using a peak-aligned CNN and whole-piece probability
averaging. The trained model is included; no training or GPU is required.
Methods, recording-held-out evaluation and limitations are in [report.pdf](report.pdf).

## Install

Use Python 3.12. From the submission folder, in Bash on Linux/macOS:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

No activation is needed. For Git Bash on Windows, replace `.venv/bin/python`
with `.venv/Scripts/python.exe` in the commands below and above.

## Prediction interface

```python
from predict import predict

labels = predict(segments, range_m, near_edge, track_id, time_s)
```

- `segments`: `(N, 5, 256)` complex64 I/Q.
- `range_m`, `near_edge`, `track_id`, `time_s`: `(N,)` arrays of float, bool, int
  and float, respectively.
- Returns N string labels in input order: `D1`, `D2`, `D3`, `D4`, `D5`, `D6`,
  `human`, `bird` or `reflector`.

Supply complete track pieces: each ID groups at most 30 observations of one target,
with no internal gap greater than 0.5 s. The model averages class probabilities
within each piece. Range and near-edge values are accepted but unused.

Inference uses an untapered Doppler FFT, peak alignment, normalized log power and
the saved CNN. The model loads once and processes segments in CPU batches.
Running `predict.py` by itself does not classify a dataset; import and call the function.

Required inference files: `predict.py`, `radar_model/`, `model/model.pt` and
`model/model_config.json`. Paths resolve relative to the code, not the working directory.

## Check a fresh environment

Use a fresh copy of the submission and the installation commands above. Place the
supplied `radar_segments.npz` at `data/radar_segments.npz`, or update the paths below.
The dataset is not included in this repository.

Call `predict()` on five real segments, treated as singleton pieces for this smoke check:

```bash
uv run --no-project --python .venv/bin/python python - <<'PY'
import numpy as np
from predict import predict

with np.load("data/radar_segments.npz", allow_pickle=False) as data:
    n = 5
    labels = predict(
        data["segments"][:n], data["range_m"][:n], data["near_edge"][:n],
        np.arange(n), data["time_s"][:n],
    )
print(labels)
PY
```

It should print five valid class names; this is not an accuracy evaluation.
For the approximately 15,000-segment CPU check, preserving complete pieces:

```bash
uv run --no-project --python .venv/bin/python python verify.py benchmark --data data/radar_segments.npz
```

Dataset loading is excluded from the reported inference timings.

Additional API checks:

```bash
uv run --no-project --python .venv/bin/python python -m unittest discover -s tests -v
```

Saved verification evidence predates the final readability edits; rerun these
checks before submission. No independent hidden-test performance is claimed.

## Optional reproduction

[research/README.md](research/README.md) documents the supplied-data requirements,
fixed recording splits, LR/XGBoost/CNN training, probability averaging and
signal-quality, range and motion diagnostics. These scripts are not used by `predict()`.
Extra dependencies are in `requirements-research.txt`; no retraining is needed to
use the included model. Generated `runs/`, environments and datasets are Git-ignored.
The Git allowlist includes only reviewed submission files; local experimental
outputs and historical evidence remain excluded.
