"""Compatibility entry point. Prefer: python ml_pipeline.py"""

from ml_pipeline import train_and_save


if __name__ == "__main__":
    _, report = train_and_save()
    print("StudentIQ model trained successfully")
    print(f"Decision threshold: {report['threshold']:.2f}")
    print(f"Confusion matrix: {report['confusion_matrix']}")
    print(f"At-risk recall: {report['at_risk_recall']:.1%}")
    print(f"False-negative rate: {report['false_negative_rate']:.1%}")
