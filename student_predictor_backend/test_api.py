"""Regression tests for the API and the supervisor's ML safety objective."""

import tempfile
import unittest
from pathlib import Path

import app as app_module


class StudentIQApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        app_module.RECORDS_PATH = Path(cls.temp_dir.name) / "student_records.json"
        app_module.app.config.update(TESTING=True)
        cls.client = app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_health_describes_at_risk_positive_class(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["positive_class"], "at_risk")

    def test_high_risk_prediction_includes_engineered_features(self):
        response = self.client.post(
            "/api/predict",
            json={
                "attendance_pct": 52,
                "homework_pct": 48,
                "midterm_score": 45,
                "study_hours_per_week": 3,
            },
        )
        result = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["predicted_outcome"], "fail")
        self.assertEqual(result["risk_flag"], "high")
        self.assertAlmostEqual(result["absence_rate"], 0.48)
        self.assertAlmostEqual(result["pass_probability"] + result["risk_probability"], 1.0)

    def test_add_and_delete_student(self):
        response = self.client.post(
            "/api/students",
            json={
                "student_name": "QA Test Student",
                "attendance_pct": 88,
                "homework_pct": 84,
                "midterm_score": 79,
                "study_hours_per_week": 10,
            },
        )
        self.assertEqual(response.status_code, 201)
        student_id = response.get_json()["student_id"]
        self.assertEqual(self.client.get(f"/api/students/{student_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/students/{student_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/students/{student_id}").status_code, 404)

    def test_validation_rejects_impossible_attendance(self):
        response = self.client.post(
            "/api/predict",
            json={
                "attendance_pct": 101,
                "homework_pct": 80,
                "midterm_score": 75,
                "study_hours_per_week": 8,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_qa_report_measures_missed_at_risk_students(self):
        report = self.client.get("/api/model-metrics").get_json()
        self.assertEqual(report["positive_class"], "at_risk")
        self.assertIn("false_negative", report["confusion_matrix"])
        self.assertGreaterEqual(report["at_risk_recall"], 0.95)


if __name__ == "__main__":
    unittest.main()
