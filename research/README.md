# Selected rectangular reproduction

Production inference never imports `research/`. The installed model is the
verified full-data peak-aligned CNN with **no taper** and whole-piece probability
averaging. The author's successful candidate checks are recorded in
`evidence/rectangular_verification.json`; the assistant did not run them.
The submission report is `../report.pdf`; editable report sources remain in the
original workspace. Saved verification predates later code readability edits.

## Data and dependencies

Use the supplied `radar_segments.npz`: 61,673 complex64 segments, shape (N,5,256),
with `task_class`, `recording_id`, `time_s`, `range_m`, `near_edge` and `label`.
Labels enter fitting/scoring only. Recording IDs enter split/piece construction
and scoring only; neither is a predictive input. Place data under ignored `data/`
or pass an external `--data` path. Never upload the supplied dataset accidentally.

From the repository root, after creating the Python 3.12 environment:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-research.txt
```

## Scripts and provenance

| File | Purpose / reused source |
|---|---|
| `make_splits.py` | Original class-aware recording allocation and validation |
| `spectral.py` | Original five features from `spectral_features.py`; shared FFT power |
| `feature_extraction.py` | Physical-study segment features and Experiment G band features |
| `classical.py` | Original `train_baselines.py` LR/XGBoost pipelines and metrics |
| `cnn.py` | Original fixed-budget CNN loop and seeded loader; optional full-data fit |
| `evaluate.py` | Explicit recording-fold evaluation, original/aligned CNN and averaging |
| `tracks.py` | Validated temporal-experiment piece construction/probability averaging |
| `signal_quality.py` | Original contrast proxy, quintiles and support/composition tables |
| `recording_diagnostics.py` | Saved development OOF recording accuracy, range comparisons and D1 5/6 errors |
| `compare_range_models.py` | Matched CNN/LR/XGBoost accuracy by range, with colored model comparisons |

Helpers are imported modules, not separate command-line programs. Entry points
are guarded by `if __name__ == "__main__"`. No archived workspace imports are
required for the commands below. FFT/power and aligned inputs are shared with
production code; rectangular FFT does not multiply I/Q by a window. It divides
squared magnitude by 256 squared. The retained five-cell denominator, -80 dB floor
and /80 log scaling are unchanged. Only rectangular preprocessing is supported;
it is the default for every entry point. The optional `--window rectangular`
argument is retained for compatibility with existing commands. Caches/checkpoints
must explicitly record rectangular preprocessing; mismatches are rejected.

The feature sets are 5/9/17/25/57/65-column prefixes: original central descriptors,
peak-relative shape, cell relationships, eight band fractions, conditional cell
fractions and spatial entropy. The 25-feature set here is G1, not temporal
statistics. Original band/secondary-peak definitions remain fixed; nothing was
retuned for rectangular inputs.

## Reproduce

The included `recording_folds.json` and `repeated_folds.json` are authoritative.
Each recording validates once per assignment, with no training/validation overlap
and all-class coverage. Optional split reconstruction writes fresh outputs:

```powershell
uv run --no-project --python .venv/Scripts/python.exe research/make_splits.py --data data/radar_segments.npz --output runs/recording_folds.json --seed 42
uv run --no-project --python .venv/Scripts/python.exe research/make_splits.py --data data/radar_segments.npz --repeated --original runs/recording_folds.json --output runs/repeated_folds.json
```

Reproduce selected comparisons from scratch; these are lengthy optional runs:

```powershell
uv run --no-project --python .venv/Scripts/python.exe research/feature_extraction.py --data data/radar_segments.npz --window rectangular --output runs/features_reproduction
uv run --no-project --python .venv/Scripts/python.exe research/evaluate.py classical --features runs/features_reproduction --window rectangular --scope all --output runs/classical_reproduction
uv run --no-project --python .venv/Scripts/python.exe research/evaluate.py cnn --data data/radar_segments.npz --window rectangular --scope all --output runs/cnn_reproduction
uv run --no-project --python .venv/Scripts/python.exe research/signal_quality.py --data data/radar_segments.npz --predictions runs/cnn_reproduction/validation_predictions.csv --window rectangular --output runs/quality_reproduction
```

Adjust data paths if relocated. Outputs must be fresh subdirectories of ignored
`runs/`; existing destinations are refused. Use `--scope development` for only the
three development folds. The robustness assignments (7/21/84) reuse recordings
and are not independent test sets. All four assignments use fixed model seed 42.

LR is L2, C=1, lbfgs, max_iter=2000 with training-only median imputation/scaling.
XGBoost uses 100 depth-three trees, lr=.1, L2=1, histogram trees and four CPU
threads, with training-only imputation. CNNs use 20 epochs, batch 256, Adam lr=.001,
weight decay=.0001, four CPU threads and no validation-controlled stopping.
Each fold starts from fresh weights/estimators.

CNN evaluation also accepts `--cnn-variants peak_aligned` for the selected model
alone. To reuse previously verified ablation folds, supply `--saved-cnn-runs`
followed by their development and robustness directories; provenance, settings,
histories and saved held-out labels are checked, without silent fallback refitting.
This optional reuse mode needs those external files; from-scratch commands do not.

Both model families save segment/local-five/whole-piece validation results,
training segment scores, fold means/SD, pooled OOF scores, class metrics,
recording accuracy, confusion matrices, probabilities and fitted fold models.
Pieces sort within recordings, split gaps >0.5 s and contain at most 30 segments.
Centered local-five windows truncate at piece edges. Whole-piece means average
nine probabilities, not labels or features, assuming one class per piece.

Report Table 5 compares **both classical models and the CNN with all three
methods**. In each run's `pooled_oof_metrics.csv`, select `assignment=development`;
for LR/XGBoost select `representation=G3` (65 features), and for the CNN select
`model=cnn_peak_aligned`. Select rows with `method=segment`, `local_five` and
`whole_track`, using `macro_f1`; these form the three report-table columns.
These are pooled OOF scores, not mean-fold scores.
`research/tracks.py` implements the same probability averaging for all models;
the class order is verified before classical probabilities are averaged.

## Recording failure diagnostics (no training or inference)

From the submission root, reuse the existing selected CNN predictions:

```powershell
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib python research/recording_diagnostics.py --data data/radar_segments.npz --output runs/recording_range_diagnostics
```

Requires `runs/cnn_rectangular/validation_predictions.csv` (not bundled in Git)
and the supplied NPZ metadata. For reproduced results, pass
`--predictions runs/cnn_reproduction/validation_predictions.csv`.
The included development split JSON must match the saved predictions.
No checkpoints or I/Q processing are needed.

Outputs under fresh `runs/recording_range_diagnostics/`:
`recording_summary.csv` (all recordings), `focus_recordings.csv` (9/18/32/68/5/6),
and `d1_prediction_counts.csv` (D1 5/6 predicted-class counts).
Also saves `accuracy_by_range.png`, `accuracy_by_range.csv` and
`accuracy_by_recording_range.csv`. The nine-panel plot uses fixed 20 m bins,
separately by class. Bars show segment-weighted accuracy, labelled as percentages.
Below each bar are its distance bin, segment count and recording count. Individual
recording-bin results remain available in `accuracy_by_recording_range.csv`.
Predictions remain whole-piece decisions, scored at each segment's own range;
a piece can contribute to several bins. No new inference or averaging is run.
Bins reuse correlated observations and recordings, so the plot is descriptive,
not a causal distance test or an independent-sample uncertainty estimate.
Segment accuracy weights observations; piece accuracy gives each at-most-30-segment
piece equal weight. Range summaries compare other same-class recording medians
and the actual same-class training recordings for each held-out fold, including
training segment range extremes. These are descriptive associations, not causal
evidence. Flight/hover status is not supplied by these files and is not inferred.
The diagnostic does not change the report or existing results.

### D1 recordings 5/6: motion indicators

Run the same diagnostic with one optional flag and a fresh output directory:

```powershell
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib python research/recording_diagnostics.py --data data/radar_segments.npz --d1-motion --output runs/d1_motion
```

This adds `d1_motion.png`/PDF (range and original central-cell peak Doppler
against time for every D1 recording), `d1_motion_summary.csv`,
`d1_motion_segments.csv`, and `d1_all_prediction_counts.csv`.
Green/orange points show correct/incorrect saved CNN decisions after whole-piece
averaging; titles highlight recordings 5/6. This is a diagnostic FFT, not model
inference. The optional flag decompresses the NPZ's full I/Q member before keeping
D1 central-cell samples; other diagnostics still use metadata only.

The range slope is a whole-recording linear trend, not instantaneous velocity.
Original peak frequencies use the existing untapered, unaligned FFT. A peak may
come from clutter or a non-body scatterer; near-zero radial motion does not prove
hovering because transverse flight may have little Doppler. No flight/hover labels
are assigned. Zero/nonfinite signals have unavailable peak frequency in the CSV.

For the same check on 65-feature LR and XGBoost, using existing saved predictions:

```powershell
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib python research/recording_diagnostics.py --data data/radar_segments.npz --predictions runs/classical_rectangular/validation_predictions.csv --model logistic_regression --output runs/range_lr
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib python research/recording_diagnostics.py --data data/radar_segments.npz --predictions runs/classical_rectangular/validation_predictions.csv --model xgboost --output runs/range_xgb
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib python research/compare_range_models.py --cnn runs/recording_range_diagnostics --lr runs/range_lr --xgb runs/range_xgb --output runs/range_model_comparison
```

`compare_range_models.py` checks identical range-bin counts and saves combined
all-class CSVs and one figure with D6/human/D1 panels. Colored bars compare
classifiers within each distance bin. Both classical models use G3 (65 features), rectangular
preprocessing and whole-piece averaging on the same development folds as the CNN.
Saved validation predictions are required; these commands never fit or run models.
The original `runs/cnn_rectangular/` and `runs/classical_rectangular/` folders are
local, not included in Git. After reproduction, pass the appropriate prediction
path from `runs/cnn_reproduction/` or `runs/classical_reproduction/` instead.

## Full-data fit and verification

```powershell
uv run --no-project --python .venv/Scripts/python.exe research/cnn.py --data data/radar_segments.npz --window rectangular --output runs/final_reproduction
uv run --no-project --python .venv/Scripts/python.exe verify_rectangular_candidate.py --model-dir runs/final_reproduction --fold-results runs/cnn_reproduction --data data/radar_segments.npz
```

Training writes weights, explicit window/classes/hash configuration and loss
history. It never replaces `model/`. The verification checks metadata, independent
FFT arithmetic, held-out probability/label parity, batching and an isolated
API copy without research code, including row ordering and CPU timing on about 15,000
segments. No accuracy on the full-data fit is presented as test performance.

## Evidence and limits

`rectangular_*` evidence files are copies of the author's completed rectangular
results and verification record. Earlier-preprocessing CSVs remain local and are
excluded from Git. They are not interchangeable with selected-window scores.
The report queries the rectangular runs, not historical metrics.

`rectangular_range_comparison.csv`, `rectangular_d1_motion.csv` and
`rectangular_d1_predictions.csv` preserve the small saved summaries supporting
the range and D1 discussion. They contain development held-out results, not new
independent tests. Per-segment OOF predictions, fold checkpoints and generated
plots are excluded; recreating diagnostics requires saved or reproduced OOF
predictions and the supplied data. No diagnostic silently retrains a model.

Signal-quality quantiles are descriptive unlabeled cohort thresholds, never
training normalization. Contrast is not calibrated SNR, and changing class/recording
composition prevents a causal interpretation of marginal accuracy trends.
Limited recording diversity and repeated exploratory use of the same groups
remain limitations; hidden-recording performance is unknown.

Not included: historical peak-scalar/shift-augmentation ablations, exploratory
BiGRU/fusion/attribution/integration studies, extended quality correlations or the
report renderer. The two unrepeated historical ablation rows were removed from the
rectangular report source. Editable report/figure sources remain in the original
workspace at `report_revision/`, using the same production FFT/feature helpers.
Use the rectangular verifier above for saved-prediction parity. `verify.py`
provides the installed CPU benchmark and package-contents inspection only.
