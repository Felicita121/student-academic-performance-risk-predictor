# Student Academic Performance & At-Risk Predictor

StudentIQ is a full-stack school dashboard that uses attendance, homework,
midterm performance and weekly study hours to predict whether a student is
likely to pass or fail. It gives staff a colour-coded student risk register,
an individual assessment tool and a transparent model QA report.

## Supervisor requirement coverage

| Role | Delivered feature | Main files |
|---|---|---|
| Frontend Engineer | Responsive school portal, named student list, high/medium/low flags, search, filters, detail view and intervention guidance | `student_predictor_frontend/` |
| Backend Developer | Validated endpoints for predictions and persistent student CRUD | `student_predictor_backend/app.py` |
| Data Engineer | Reusable pipeline deriving absence ratio, weighted performance, study efficiency and low engagement | `student_predictor_backend/ml_pipeline.py` |
| ML Core Modeler | Standardised Logistic Regression binary classifier with an explicit at-risk positive class | `ml_pipeline.py`, `model.pkl` |
| QA & Analyst | Five-fold stratified cross-validation, threshold selection and visible confusion matrix focused on false negatives | `cross_validation.py`, `model_metrics.json`, Model QA page |

## Important ML correction

`at_risk` is encoded as the positive class. Therefore, a **false negative**
now has the correct operational meaning: a student who is actually at risk but
was not flagged. The earlier implementation encoded `pass` as positive and
lowered the pass threshold, which did not correctly support the stated safety
objective.

The bundled model uses a threshold selected from out-of-fold predictions. On
the supplied 100-row teaching dataset it records:

- 99% cross-validated accuracy
- 100% recall for at-risk students
- 0 false negatives (missed at-risk students)
- 1 false positive

These results reflect a small, highly structured teaching dataset and should
not be presented as proof of real-world generalisation.

## Run locally

Backend:

```bash
cd student_predictor_backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
.venv/Scripts/python ml_pipeline.py              # retrain + evaluate
.venv/Scripts/python app.py
```

Frontend, in a second terminal:

```bash
cd student_predictor_frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`. The frontend API URL is configured in
`student_predictor_frontend/config.js` and can also be changed from the UI.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service, model and threshold status |
| GET | `/api/students` | Risk-prioritised student register |
| GET | `/api/students/<id>` | One student record |
| POST | `/api/students` | Validate, engineer features, predict and persist |
| DELETE | `/api/students/<id>` | Delete a stored student |
| POST | `/api/predict` | What-if prediction without saving |
| GET | `/api/model-metrics` | Cross-validation metrics and confusion matrix |
| GET | `/api/feature-engineering` | Feature definitions used by the pipeline |

Example prediction body:

```json
{
  "student_name": "Alex Morgan",
  "attendance_pct": 72,
  "homework_pct": 68,
  "midterm_score": 61,
  "study_hours_per_week": 7
}
```

## Verification

```bash
cd student_predictor_backend
python -m unittest -v test_api.py
python cross_validation.py
```

## Deployment

The GitHub Actions workflow deploys `student_predictor_frontend` to GitHub
Pages. Deploy `student_predictor_backend` as a Python web service with:

```text
gunicorn app:app
```

Set `ALLOWED_ORIGINS` to the public frontend origin and update `config.js` with
the public `/api` base URL. Added records are written to
`student_records.json`; for durable cloud storage, set `STUDENT_DATA_PATH` to a
persistent disk path.
