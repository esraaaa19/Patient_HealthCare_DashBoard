"""Reproduces the notebook's FINAL model: Gradient Boosting, patient-level
80/20 split, threshold 0.1176 chosen on validation data."""
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from features import load_clean, NUMERIC, CATEGORICAL, FEATURES

THRESHOLD = 0.1176          # selected on the validation set in the notebook

df = load_clean()
X, y = df[FEATURES], df["readmitted_30"]

# patient-level split: no patient appears in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["patient_nbr"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print("Train:", X_train.shape, " Test:", X_test.shape)

pre = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
])
model = Pipeline([
    ("pre", pre),
    ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)),
])
print("Training Gradient Boosting (about 1-2 minutes)...")
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= THRESHOLD).astype(int)

print("\nFinal test results")
print("Threshold :", THRESHOLD)
print("Accuracy  :", round(accuracy_score(y_test, pred), 4))
print("Precision :", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall    :", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1 Score  :", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC   :", round(roc_auc_score(y_test, prob), 4))
print("PR-AUC    :", round(average_precision_score(y_test, prob), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nNotebook reported: Precision 0.1644, Recall 0.4775, F1 0.2446, "
      "ROC-AUC 0.6389, PR-AUC 0.1825, confusion [[12781 5221] [1124 1027]]")

# typical-patient defaults (used by the Risk Estimator for fields it doesn't ask)
defaults = {c: float(X_train[c].median()) for c in NUMERIC}
defaults.update({c: X_train[c].mode()[0] for c in CATEGORICAL})

names = pd.Index(model.named_steps["pre"].get_feature_names_out())
names = names.str.replace(r"^(num|cat)__", "", regex=True)
importance = (pd.Series(model.named_steps["clf"].feature_importances_, index=names)
              .sort_values(ascending=False).head(15))

joblib.dump({"model": model, "threshold": THRESHOLD, "defaults": defaults,
             "y_test": y_test.values, "prob_test": prob,
             "importance": importance}, "model.joblib")
print("\nSaved model.joblib")
