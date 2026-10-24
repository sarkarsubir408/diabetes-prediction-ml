"""Train and evaluate an educational diabetes-classification model.

Expected input: diabetes.csv, with the Pima-style columns documented in README.md.
This is a learning demo, not a medical diagnostic tool.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diabetes.csv"
MODEL_PATH = ROOT / "diabetes_model.joblib"
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# In the commonly used Pima dataset, zero is not a plausible measurement for
# these fields and is conventionally treated as a missing-value placeholder.
# Do NOT include Pregnancies: zero pregnancies is a valid value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload the dataset described in README.md "
            "and save it as diabetes.csv in the project directory."
        )

    data = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data[FEATURES + [TARGET]].copy()
    if data[TARGET].isna().any() or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only non-missing binary labels 0 and 1.")

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X, y


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"knn__n_neighbors": list(range(1, 50))},
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # The test set is used only after model selection is complete.
    predictions = model.predict(X_test)
    print(f"Rows: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Best k: {search.best_params_['knn__n_neighbors']}")
    print(f"Best 5-fold CV recall: {search.best_score_:.3f}")
    print("\nHeld-out test metrics (educational evaluation only)")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))

    # Save the complete fitted pipeline and feature metadata together.
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nSaved model bundle to: {MODEL_PATH}")
    print("Do not interpret this model output as a medical diagnosis or calibrated risk.")


if __name__ == "__main__":
    main()
