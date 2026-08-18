"""StudentIQ Flask API: records, engineered features and risk predictions."""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from threading import RLock

import pandas as pd
from flask import Flask, jsonify, request

from ml_pipeline import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_INPUT_FIELDS,
    engineer_features,
    train_and_save,
)


BASE_DIR = Path(__file__).resolve().parent
RECORDS_PATH = Path(os.environ.get("STUDENT_DATA_PATH", BASE_DIR / "student_records.json"))
LOCK = RLock()


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        train_and_save()
    with MODEL_PATH.open("rb") as handle:
        loaded = pickle.load(handle)
    # Reject the earlier bundle whose positive class and threshold semantics
    # were incompatible with the supervisor's false-negative objective.
    if loaded.get("positive_class") != "at_risk":
        loaded, _ = train_and_save()
    return loaded


BUNDLE = load_bundle()
MODEL = BUNDLE["model"]
FEATURE_COLS = BUNDLE["features"]
THRESHOLD = float(BUNDLE["threshold"])

app = Flask(__name__)


def validate_inputs(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("A JSON object is required.")
    missing = [field for field in RAW_INPUT_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    try:
        record = {field: float(data[field]) for field in RAW_INPUT_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError("All academic indicators must be numeric.") from exc

    for field in ("attendance_pct", "homework_pct", "midterm_score"):
        if not 0 <= record[field] <= 100:
            raise ValueError(f"{field} must be between 0 and 100.")
    if not 0 <= record["study_hours_per_week"] <= 80:
        raise ValueError("study_hours_per_week must be between 0 and 80.")
    return record


def score_record(record: dict) -> dict:
    enriched = {**record, **engineer_features(record)}
    frame = pd.DataFrame(
        [[enriched[column] for column in FEATURE_COLS]], columns=FEATURE_COLS
    )
    risk_probability = float(MODEL.predict_proba(frame)[0][1])
    predicted_at_risk = risk_probability >= THRESHOLD
    medium_boundary = max(0.20, THRESHOLD * 0.55)
    if predicted_at_risk:
        risk_flag = "high"
    elif risk_probability >= medium_boundary:
        risk_flag = "medium"
    else:
        risk_flag = "low"

    enriched.update(
        {
            "prediction": 0 if predicted_at_risk else 1,
            "predicted_outcome": "fail" if predicted_at_risk else "pass",
            "risk_probability": round(risk_probability, 4),
            "pass_probability": round(1.0 - risk_probability, 4),
            "risk_flag": risk_flag,
            "recommendation": recommendation_for(risk_flag),
        }
    )
    return enriched


def recommendation_for(risk_flag: str) -> str:
    if risk_flag == "high":
        return "Prioritise an adviser check-in and an academic support plan."
    if risk_flag == "medium":
        return "Monitor weekly and offer targeted study or attendance support."
    return "Continue routine monitoring and reinforce current study habits."


def seed_students() -> dict[int, dict]:
    records: dict[int, dict] = {}
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            student_id = int(row["student_id"])
            base = {field: float(row[field]) for field in RAW_INPUT_FIELDS}
            base.update(
                {
                    "student_id": student_id,
                    "student_name": f"Student {student_id:03d}",
                    "actual_outcome": "pass" if int(float(row["pass"])) else "fail",
                    "record_source": "seed",
                }
            )
            records[student_id] = score_record(base)
    return records


def load_students() -> dict[int, dict]:
    if not RECORDS_PATH.exists():
        return seed_students()
    try:
        saved = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
        return {int(item["student_id"]): score_record(item) for item in saved}
    except (OSError, ValueError, KeyError, TypeError):
        return seed_students()


students = load_students()
next_id = max(students, default=0) + 1


def persist_students() -> None:
    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECORDS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(list(students.values()), indent=2), encoding="utf-8"
    )
    os.replace(temporary, RECORDS_PATH)


def ordered_students() -> list[dict]:
    return sorted(
        students.values(),
        key=lambda item: (-float(item["risk_probability"]), int(item["student_id"])),
    )


def model_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        **BUNDLE.get("metrics", {}),
        "model_type": BUNDLE.get("model_type", "Logistic Regression"),
        "positive_class": "at_risk",
    }


ALLOWED_ORIGINS = [
    value.strip()
    for value in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if value.strip()
]


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if ALLOWED_ORIGINS == ["*"]:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "students_loaded": len(students),
            "model_type": BUNDLE.get("model_type", "Logistic Regression"),
            "model_version": BUNDLE.get("model_version", "2.0"),
            "positive_class": "at_risk",
            "decision_threshold": THRESHOLD,
        }
    )


@app.route("/api/students", methods=["GET"])
def get_students():
    return jsonify(ordered_students())


@app.route("/api/students/<int:student_id>", methods=["GET"])
def get_student(student_id: int):
    student = students.get(student_id)
    if student is None:
        return jsonify({"error": "Student not found."}), 404
    return jsonify(student)


@app.route("/api/students", methods=["POST"])
def add_student():
    global next_id
    data = request.get_json(silent=True)
    try:
        record = validate_inputs(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with LOCK:
        requested_id = data.get("student_id")
        if requested_id is None:
            student_id = next_id
        else:
            try:
                student_id = int(requested_id)
            except (TypeError, ValueError):
                return jsonify({"error": "student_id must be an integer."}), 400
            if student_id <= 0:
                return jsonify({"error": "student_id must be positive."}), 400
            if student_id in students:
                return jsonify({"error": "student_id already exists."}), 409

        name = str(data.get("student_name", "")).strip() or f"Student {student_id:03d}"
        if len(name) > 80:
            return jsonify({"error": "student_name must be 80 characters or fewer."}), 400

        record.update(
            {
                "student_id": student_id,
                "student_name": name,
                "record_source": "entered",
            }
        )
        actual = str(data.get("actual_outcome", "")).lower().strip()
        if actual:
            if actual not in {"pass", "fail"}:
                return jsonify({"error": "actual_outcome must be pass or fail."}), 400
            record["actual_outcome"] = actual

        saved = score_record(record)
        students[student_id] = saved
        next_id = max(next_id, student_id + 1)
        persist_students()
    return jsonify(saved), 201


@app.route("/api/predict", methods=["POST"])
def predict_only():
    data = request.get_json(silent=True)
    try:
        record = validate_inputs(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result = score_record(record)
    return jsonify(result)


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id: int):
    with LOCK:
        if student_id not in students:
            return jsonify({"error": "Student not found."}), 404
        del students[student_id]
        persist_students()
    return jsonify({"message": "Student deleted.", "student_id": student_id})


@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    return jsonify(model_metrics())


@app.route("/api/feature-engineering", methods=["GET"])
def get_feature_engineering():
    return jsonify(
        {
            "raw_inputs": RAW_INPUT_FIELDS,
            "engineered_features": {
                "absence_rate": "(100 - attendance_pct) / 100",
                "performance_score": "40% midterm + 30% homework + 30% attendance",
                "study_efficiency": "midterm_score / max(study_hours_per_week, 1)",
                "low_engagement_flag": "attendance < 70% or homework < 60%",
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=port,
    )
