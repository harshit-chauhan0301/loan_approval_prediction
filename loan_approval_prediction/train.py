"""
train.py - Train, compare and save the best Loan Approval model.

Run:  python train.py

Steps:
  1. Load data/loan_prediction.csv
  2. Split 80:20 (random_state=42, stratified)
  3. Build one preprocessing + model Pipeline per algorithm
     (preprocessing is fitted ONLY on training data -> no data leakage)
  4. Evaluate on the test set and compare the models
  5. Pick the best model by cross-validated F1-score (on training data only)
  6. Save the best pipeline to models/best_model.pkl
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_prediction.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

RANDOM_STATE = 42
TARGET = "Loan_Status"

# Feature groups. app.py uses the same column names and order.
NUMERICAL_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
# Credit_History is a 0/1 flag, so it is treated as a category (no scaling needed).
CATEGORICAL_FEATURES = ["Gender", "Married", "Dependents", "Education",
                        "Self_Employed", "Credit_History", "Property_Area"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            "Download a Loan Prediction CSV and save it as data/loan_prediction.csv (see README.md)."
        )
    df = pd.read_csv(DATA_PATH)
    needed = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot for categories."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" -> unseen categories never crash prediction
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models() -> dict:
    """Three full pipelines (same preprocessing, different classifier)."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        for name, clf in classifiers.items()
    }


def evaluate(name, model, X_test, y_test) -> dict:
    """Print the detailed report and return a dict of metrics."""
    y_pred = model.predict(X_test)
    print("\n" + "-" * 60)
    print(f"{name}")
    print("-" * 60)
    print("Confusion Matrix (rows = actual, columns = predicted) [Rejected, Approved]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"],
                                zero_division=0))
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }


def plot_comparison(results: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    melted = results.melt(id_vars="Model", value_vars=metrics,
                          var_name="Metric", value_name="Score")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (test set)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)
    print("Saved -> outputs/model_comparison.png")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    # Drop rows with a missing target and map Y/N -> 1/0
    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].map({"Y": 1, "N": 0})
    if y.isnull().any():
        print("ERROR: Loan_Status must only contain 'Y' or 'N'.")
        sys.exit(1)
    y = y.astype(int)

    # 80:20 split; stratify keeps the approval ratio similar in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, pipeline in models.items():
        # Cross-validation uses ONLY the training data (preprocessing is refit per fold)
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        # Final fit: imputers, scaler and encoder learn from training data only
        pipeline.fit(X_train, y_train)
        metrics = evaluate(name, pipeline, X_test, y_test)
        metrics["CV F1 (train)"] = cv_f1.mean()
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results.round(3).to_string())
    plot_comparison(results.reset_index())

    # Selection criterion: highest cross-validated F1-score on the TRAINING data.
    # F1 balances precision and recall, and choosing on training-set CV keeps the
    # test set untouched for an honest final evaluation. Ties are broken by test F1.
    ranked = results.sort_values(["CV F1 (train)", "F1-score"], ascending=False)
    best_name = ranked.index[0]
    print(f"\nBest model: {best_name} "
          f"(CV F1 = {results.loc[best_name, 'CV F1 (train)']:.3f}, "
          f"test F1 = {results.loc[best_name, 'F1-score']:.3f})")

    joblib.dump(models[best_name], MODEL_PATH)
    print(f"Saved best pipeline -> {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
