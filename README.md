# Student Academic Performance & At-Risk Predictor

A full-stack academic analytics project that uses a trained machine-learning model to estimate student pass probability and flag academic risk.

## Project structure

```text
student_academic_performance_risk/
├── student_predictor_backend/
│   ├── app.py
│   ├── model.pkl
│   ├── features.csv
│   ├── requirements.txt
│   ├── Procfile
│   ├── Trainmodel.py
│   └── cross_validation.py
└── student_predictor_frontend/
    ├── index.html
    ├── style.css
    ├── script.js
    ├── config.js
    └── README.md
```

## Local development

Backend:

```bash
cd student_predictor_backend
pip install -r requirements.txt
python app.py
```

Frontend (second terminal):

```bash
cd student_predictor_frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`.

The frontend talks to `http://127.0.0.1:5000/api` by default.

## API

- `GET /api/health`
- `GET /api/students`
- `GET /api/students/<id>`
- `POST /api/students`
- `POST /api/predict`
- `DELETE /api/students/<id>`

## Deployment preparation

The frontend and backend are deliberately separated so they can be deployed independently. Before deployment, update `student_predictor_frontend/config.js` with the public backend API URL. The backend already includes CORS support through `ALLOWED_ORIGINS`.

Keep `model.pkl` and `features.csv` alongside `app.py` in the backend deployment directory.
