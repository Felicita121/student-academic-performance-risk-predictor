"""
Student Academic Performance & At-Risk Predictor - Backend API
Role: Backend Developer

Endpoints:
  GET  /api/students            -> list all students with risk predictions
  GET  /api/students/<id>       -> get one student
  POST /api/students            -> add a new student data point (predicts + stores)
  POST /api/predict             -> predict only, don't store (for a "what-if" check)
  DELETE /api/students/<id>     -> remove a student
"""

import pickle
import csv
import os
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'features.csv')

# ---- Load trained model, scaler, feature list, threshold ----
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
MODEL = bundle['model']
SCALER = bundle['scaler']
FEATURE_COLS = bundle['features']
THRESHOLD = bundle['threshold']

RAW_INPUT_FIELDS = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']

students = {}
next_id = 1


def load_seed_data():
    global next_id
    if not os.path.exists(DATA_PATH):
        return
    with open(DATA_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['student_id'])
            record = {k: float(row[k]) for k in RAW_INPUT_FIELDS}
            record['student_id'] = sid
            record['actual_pass'] = int(float(row['pass']))
            derived = engineer_features(record)
            record.update(derived)
            record['prediction'], record['risk_probability'], record['risk_flag'] = predict(record)
            students[sid] = record
            next_id = max(next_id, sid + 1)


def engineer_features(record):
    """Data Engineer role logic, reapplied to a single new record."""
    attendance = record['attendance_pct']
    homework = record['homework_pct']
    midterm = record['midterm_score']
    study_hours = record['study_hours_per_week'] or 1

    absence_rate = (100 - attendance) / 100
    performance_score = 0.4 * midterm + 0.3 * homework + 0.3 * attendance
    study_efficiency = midterm / study_hours
    low_engagement_flag = 1 if (attendance < 70 or homework < 60) else 0

    return {
        'absence_rate': absence_rate,
        'performance_score': performance_score,
        'study_efficiency': study_efficiency,
        'low_engagement_flag': low_engagement_flag,
    }

def predict(record):
    """Run the ML Core Modeler's Logistic Regression model on one record."""
    X = pd.DataFrame([[record[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    X_scaled = SCALER.transform(X)
    prob_pass = MODEL.predict_proba(X_scaled)[0][1]
    prediction = 1 if prob_pass >= THRESHOLD else 0
    if prob_pass < THRESHOLD:
        risk_flag = 'high'
    elif prob_pass < 0.65:
        risk_flag = 'medium'
    else:
        risk_flag = 'low'
    return prediction, round(float(prob_pass), 4), risk_flag

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()
]


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if ALLOWED_ORIGINS == ['*']:
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api/students', methods=['GET'])
def get_students():
    return jsonify(list(students.values()))


@app.route('/api/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    student = students.get(student_id)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(student)


@app.route('/api/students', methods=['POST'])
def add_student():
    global next_id
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    missing = [f for f in RAW_INPUT_FIELDS if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        record = {f: float(data[f]) for f in RAW_INPUT_FIELDS}
    except (TypeError, ValueError):
        return jsonify({'error': 'All fields must be numeric'}), 400

    if not (0 <= record['attendance_pct'] <= 100):
        return jsonify({'error': 'attendance_pct must be between 0 and 100'}), 400
    if not (0 <= record['homework_pct'] <= 100):
        return jsonify({'error': 'homework_pct must be between 0 and 100'}), 400
    if not (0 <= record['midterm_score'] <= 100):
        return jsonify({'error': 'midterm_score must be between 0 and 100'}), 400
    if record['study_hours_per_week'] < 0:
        return jsonify({'error': 'study_hours_per_week cannot be negative'}), 400

    student_id = data.get('student_id', next_id)
    record['student_id'] = student_id
    record.update(engineer_features(record))
    record['prediction'], record['risk_probability'], record['risk_flag'] = predict(record)

    students[student_id] = record
    next_id = max(next_id, student_id + 1)

    return jsonify(record), 201


@app.route('/api/predict', methods=['POST'])
def predict_only():
    """What-if endpoint: run a prediction without saving the student."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    missing = [f for f in RAW_INPUT_FIELDS if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        record = {f: float(data[f]) for f in RAW_INPUT_FIELDS}
    except (TypeError, ValueError):
        return jsonify({'error': 'All fields must be numeric'}), 400

    record.update(engineer_features(record))
    prediction, prob, risk_flag = predict(record)

    return jsonify({
        'prediction': prediction,
        'risk_probability': prob,
        'risk_flag': risk_flag,
    })

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    if student_id not in students:
        return jsonify({'error': 'Student not found'}), 404
    del students[student_id]
    return jsonify({'message': 'Deleted'}), 200


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'students_loaded': len(students), 'threshold': THRESHOLD})


load_seed_data()
print(f"Loaded {len(students)} students. Threshold = {THRESHOLD}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
