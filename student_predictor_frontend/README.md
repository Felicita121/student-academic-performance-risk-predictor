# StudentIQ — Academic Risk Predictor Frontend

Modern vanilla HTML/CSS/JavaScript frontend for the supplied Flask API.

## Run locally

1. Start the backend from `student_predictor_backend`:

```bash
python app.py
```

2. From `student_predictor_frontend`, serve the static frontend:

```bash
python -m http.server 5500
```

3. Open `http://127.0.0.1:5500`.

The default API URL is stored in `config.js`:

```js
window.APP_CONFIG = {
  API_BASE: "http://127.0.0.1:5000/api"
};
```

The UI also has **API Settings** in the sidebar, and the selected URL is stored in browser local storage for convenience.

## Features

- Modern responsive dashboard
- Risk distribution visualization
- Class performance averages
- What-if prediction using `POST /api/predict`
- Save prediction as a student using `POST /api/students`
- Student search and risk filters
- Student detail modal
- Delete saved students
- Analytics overview
- Model QA page with at-risk recall and confusion matrix
- Named student records and intervention recommendations
- Read-only demo data fallback when the hosted API is waking or unavailable
- API health/status indicator
- Local/deployment API URL configuration

The frontend does not modify the model or the prediction logic.
