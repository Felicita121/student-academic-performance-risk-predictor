"""Shared feature engineering, training and evaluation for StudentIQ.

The positive class is deliberately ``at_risk``. This makes a false negative
mean exactly what the school cares about: a student who is actually at risk
but was not flagged by the model.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "features.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

RAW_INPUT_FIELDS = [
    "attendance_pct",
    "homework_pct",
    "midterm_score",
    "study_hours_per_week",
]

FEATURE_COLS = [
    *RAW_INPUT_FIELDS,
    "absence_rate",
    "performance_score",
    "study_efficiency",
    "low_engagement_flag",
]


def engineer_features(record: dict) -> dict:
    """Create analytical features from one aggregated student record."""
    attendance = float(record["attendance_pct"])
    homework = float(record["homework_pct"])
    midterm = float(record["midterm_score"])
    study_hours = float(record["study_hours_per_week"])

    return {
        # Ratio of absences to total classes, derived from attendance rate.
        "absence_rate": round((100.0 - attendance) / 100.0, 4),
        "performance_score": round(
            0.40 * midterm + 0.30 * homework + 0.30 * attendance, 4
        ),
        "study_efficiency": round(midterm / max(study_hours, 1.0), 4),
        "low_engagement_flag": int(attendance < 70.0 or homework < 60.0),
    }


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    # labels=[0, 1] => 0 is not-at-risk/pass, 1 is at-risk/fail.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": round(float(threshold), 2),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "at_risk_recall": round(recall, 4),
        "at_risk_precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "false_negative_rate": round(1.0 - recall, 4),
        "students_evaluated": total,
    }


def cross_validated_probabilities(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Return out-of-fold probabilities so every row is evaluated unseen."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in folds.split(X, y):
        model = _pipeline()
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return probabilities


def choose_threshold(y_true: np.ndarray, risk_probabilities: np.ndarray) -> tuple[float, dict]:
    """Choose a useful threshold while prioritising at-risk recall.

    We first require at least 95% recall for at-risk students, then maximise
    balanced accuracy. This avoids the meaningless solution of flagging every
    student while still explicitly minimising missed at-risk students.
    """
    candidates = []
    for threshold in np.arange(0.20, 0.81, 0.01):
        predicted = (risk_probabilities >= threshold).astype(int)
        result = _metrics(y_true, predicted, float(threshold))
        balanced_accuracy = (result["at_risk_recall"] + result["specificity"]) / 2
        candidates.append((result, balanced_accuracy))

    eligible = [item for item in candidates if item[0]["at_risk_recall"] >= 0.95]
    pool = eligible or candidates
    best, _ = max(
        pool,
        key=lambda item: (
            item[1],
            item[0]["at_risk_recall"],
            item[0]["at_risk_precision"],
            item[0]["threshold"],
        ),
    )
    return float(best["threshold"]), best


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[dict, dict]:
    frame = pd.read_csv(data_path)
    X = frame[FEATURE_COLS]
    # at_risk=1 is the positive class; fail/pass in the source remains intact.
    y = (1 - frame["pass"].astype(int)).to_numpy()

    oof_probabilities = cross_validated_probabilities(X, y)
    threshold, metrics = choose_threshold(y, oof_probabilities)

    model = _pipeline()
    model.fit(X, y)
    bundle = {
        "model": model,
        "features": FEATURE_COLS,
        "threshold": threshold,
        "positive_class": "at_risk",
        "model_type": "Logistic Regression",
        "model_version": "2.0",
        "metrics": metrics,
    }
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    report = {
        **metrics,
        "model_type": bundle["model_type"],
        "model_version": bundle["model_version"],
        "positive_class": bundle["positive_class"],
        "evaluation_method": "5-fold stratified cross-validation",
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "objective": "Minimise false negatives for at-risk students",
        "dataset_note": (
            "The supplied 100-row teaching dataset is small and highly structured. "
            "Cross-validation results should not be treated as proof of real-world performance."
        ),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return bundle, report


if __name__ == "__main__":
    _, saved_report = train_and_save()
    print(json.dumps(saved_report, indent=2))
