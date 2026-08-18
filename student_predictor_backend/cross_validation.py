"""QA report for the at-risk classifier using unseen out-of-fold predictions."""

import json

from ml_pipeline import train_and_save


if __name__ == "__main__":
    _, report = train_and_save()
    print("=== StudentIQ QA & Analyst Report ===")
    print(json.dumps(report, indent=2))
