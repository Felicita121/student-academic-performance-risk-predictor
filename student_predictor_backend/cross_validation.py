import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv('features.csv')
feature_cols = ['attendance_pct', 'homework_pct', 'midterm_score',
                 'study_hours_per_week', 'absence_rate', 'performance_score',
                 'study_efficiency', 'low_engagement_flag']
X = df[feature_cols].values
y = df['pass'].values

threshold = 0.40
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_tn, all_fp, all_fn, all_tp = 0,0,0,0
fold_accs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:,1]
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    all_tn += tn; all_fp += fp; all_fn += fn; all_tp += tp
    acc = (tn+tp)/(tn+fp+fn+tp)
    fold_accs.append(acc)
    print(f"Fold {fold}: n_test={len(y_test)}  TN={tn} FP={fp} FN={fn} TP={tp}  acc={acc:.2%}")

print("\n=== Aggregated over 5 folds (all 100 students used as test exactly once) ===")
print(f"TN={all_tn} FP={all_fp} FN={all_fn} TP={all_tp}")
total = all_tn+all_fp+all_fn+all_tp
acc = (all_tn+all_tp)/total
fnr = all_fn/(all_fn+all_tp) if (all_fn+all_tp)>0 else 0
precision = all_tp/(all_tp+all_fp) if (all_tp+all_fp)>0 else 0
recall = all_tp/(all_tp+all_fn) if (all_tp+all_fn)>0 else 0
print(f"Accuracy: {acc:.2%}")
print(f"False Negative Rate (missed at-risk): {fnr:.2%}")
print(f"Precision (Pass): {precision:.2%}")
print(f"Recall (Pass): {recall:.2%}")
print(f"\nMean fold accuracy: {np.mean(fold_accs):.2%}  (std: {np.std(fold_accs):.2%})")