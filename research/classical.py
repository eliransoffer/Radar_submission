"""Original fixed LR/XGBoost factories and metric tables; called by evaluate.py."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from radar_model.preprocessing import CLASSES
CLASS_INDICES = np.arange(len(CLASSES))

def make_models():
    """Fresh estimators per fold; only training data reaches fit()."""
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            # lbfgs with default penalty uses L2 regularization; C fixes strength.
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
        ),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            # Trees need no standardization. Depth 3 limits each tree's complexity.
            XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                objective="multi:softprob", num_class=len(CLASSES),
                tree_method="hist", device="cpu", n_jobs=4,
                reg_lambda=1.0, random_state=42, eval_metric="mlogloss",
            ),
        ),
    }

def evaluate_predictions(targets, predictions, recording_ids, model_name, fold, split, output):
    """Return segment metrics, per-class metrics, and recording accuracies."""
    if len(predictions) != len(targets) or not np.isin(predictions, CLASS_INDICES).all():
        raise ValueError("Expected one valid class prediction per input segment.")
    identifiers = {"model": model_name, "fold": fold, "split": split}
    metrics = {
        **identifiers,
        "n_segments": len(targets),
        "n_recordings": len(np.unique(recording_ids)),
        "macro_f1": f1_score(targets, predictions, labels=CLASS_INDICES,
                             average="macro", zero_division=0),
        "accuracy": accuracy_score(targets, predictions),
    }
    precision, recalls, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=CLASS_INDICES, zero_division=0
    )
    class_rows = []
    for index, class_name in enumerate(CLASSES):
        row = identifiers.copy()
        row["class"] = class_name
        row["precision"] = precision[index]
        row["recall"] = recalls[index]
        row["f1"] = f1[index]
        row["n_segments"] = int(support[index])
        class_rows.append(row)
    recording_rows = []
    for recording_id in np.unique(recording_ids):
        mask = recording_ids == recording_id
        recording_rows.append({
            **identifiers, "recording_id": int(recording_id),
            "class": CLASSES[int(targets[mask][0])], "n_segments": int(mask.sum()),
            "accuracy": accuracy_score(targets[mask], predictions[mask]),
        })

    # Rows are true classes; columns are predicted classes, always in CLASSES order.
    matrix = confusion_matrix(targets, predictions, labels=CLASS_INDICES)
    counts = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    counts.index.name = "true_class"
    counts.to_csv(output / f"{model_name}_fold{fold}_{split}_confusion_counts.csv")
    row_totals = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                          where=row_totals > 0)
    normalized = pd.DataFrame(fractions, index=CLASSES, columns=CLASSES)
    normalized.index.name = "true_class"
    normalized.to_csv(output / f"{model_name}_fold{fold}_{split}_confusion_normalized.csv")
    return metrics, class_rows, recording_rows
