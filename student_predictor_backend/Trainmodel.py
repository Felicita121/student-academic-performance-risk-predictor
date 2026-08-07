import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pickle

df = pd.read_csv('features.csv')

feature_cols = ['attendance_pct', 'homework_pct', 'midterm_score',
                 'study_hours_per_week', 'absence_rate', 'performance_score',
                 'study_efficiency', 'low_engagement_flag']

X = df[feature_cols]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 0.40 threshold as established: flag as "pass" only if prob >= 0.40,
# otherwise flag as at-risk (0) -> biases toward catching more at-risk students
probs = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.40
y_pred = (probs >= threshold).astype(int)

# Also standard 0.5 for comparison
y_pred_default = (probs >= 0.5).astype(int)

print("=== Test set size:", len(y_test), "===\n")

print("--- Confusion Matrix @ 0.40 threshold ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Neg: {tn}, False Pos: {fp}, False Neg: {fn}, True Pos: {tp}")
print(f"False Negative Rate (missed at-risk students): {fn/(fn+tp):.2%}" if (fn+tp)>0 else "N/A")
print(classification_report(y_test, y_pred, target_names=['At-Risk(0)','Pass(1)']))

print("\n--- Confusion Matrix @ 0.50 threshold (default, for comparison) ---")
cm2 = confusion_matrix(y_test, y_pred_default)
print(cm2)
tn2, fp2, fn2, tp2 = cm2.ravel()
print(f"False Negative Rate: {fn2/(fn2+tp2):.2%}" if (fn2+tp2)>0 else "N/A")

# Save model + scaler + feature list
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'features': feature_cols, 'threshold': threshold}, f)

print("\nModel coefficients:")
for feat, coef in zip(feature_cols, model.coef_[0]):
    print(f"  {feat}: {coef:.4f}")