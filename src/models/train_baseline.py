from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/training_dataset.csv"
MODEL_PATH = ROOT / "models/logistic_regression.joblib"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

target = "route_hazard_next_7_days"

exclude = {
    target,
    "date",
    "label_source",
    "geo_hazard_day",
    "geo_hazard_event_count",
    "geo_security_events_7d",
    "geo_hostile_events_7d",
    "geo_avg_tone_7d",
    "geo_risk_score",
    "archive_available",
}

X = df.drop(columns=[c for c in exclude if c in df.columns])
y = df[target].astype(int)

split = int(len(df) * 0.8)

X_train = X.iloc[:split - 7]
y_train = y.iloc[:split - 7]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

numeric_features = X.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_features,
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Features:", X.columns.tolist())
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Last training date:", df["date"].iloc[split - 8].date())
print("First testing date:", df["date"].iloc[split].date())
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

if y_test.nunique() == 2:
    print("PR-AUC:", round(average_precision_score(y_test, probabilities), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
else:
    print("Test set contains only one class; AUC metrics unavailable.")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Saved model:", MODEL_PATH)
