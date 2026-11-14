"""
eda.py - Exploratory Data Analysis for the Loan Approval Prediction System.

Run:  python eda.py
Plots are saved inside the outputs/ folder.
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_prediction.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

REQUIRED_COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Status",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and check that it has the expected columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download a Loan Prediction CSV (columns: " + ", ".join(REQUIRED_COLUMNS) + ")\n"
            "and save it as data/loan_prediction.csv (see README.md)."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure into outputs/ and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved -> {path.relative_to(BASE_DIR)}")


def print_overview(df: pd.DataFrame) -> None:
    """Print shape, info, statistics and missing values."""
    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    print("=" * 60)
    print("STATISTICAL SUMMARY (numerical columns)")
    print("=" * 60)
    print(df.describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    missing = df.isnull().sum()
    report = pd.DataFrame({
        "Missing": missing,
        "Percent": (missing / len(df) * 100).round(2),
    })
    print(report[report["Missing"] > 0].sort_values("Missing", ascending=False).to_string())


def plot_loan_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Y", "N"]
    sns.countplot(data=df, x="Loan_Status", order=order, hue="Loan_Status",
                  hue_order=order, palette={"Y": "#2e8b57", "N": "#c0392b"},
                  legend=False, ax=ax)
    ax.set_title("Loan Approval Distribution")
    ax.set_xlabel("Loan Status (Y = Approved, N = Rejected)")
    ax.set_ylabel("Number of applications")
    for container in ax.containers:
        ax.bar_label(container)
    save_figure(fig, "loan_distribution.png")


def plot_income_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["ApplicantIncome"], bins=40, kde=True, color="#2c7fb8", ax=ax)
    ax.set_title("Applicant Income Distribution")
    ax.set_xlabel("Applicant Income")
    save_figure(fig, "applicant_income_distribution.png")


def plot_loan_amount_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["LoanAmount"].dropna(), bins=40, kde=True, color="#7b3294", ax=ax)
    ax.set_title("Loan Amount Distribution")
    ax.set_xlabel("Loan Amount (in thousands)")
    save_figure(fig, "loan_amount_distribution.png")


def plot_category_vs_status(df: pd.DataFrame, column: str, filename: str) -> None:
    """Count plot of a categorical column split by loan status."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df.dropna(subset=[column]).copy()
    data[column] = data[column].astype(str)
    sns.countplot(data=data, x=column, hue="Loan_Status", hue_order=["Y", "N"],
                  palette={"Y": "#2e8b57", "N": "#c0392b"}, ax=ax)
    ax.set_title(f"{column.replace('_', ' ')} vs Loan Approval")
    ax.set_ylabel("Number of applications")
    save_figure(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                    "Loan_Amount_Term", "Credit_History"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numerical columns)")
    save_figure(fig, "correlation_heatmap.png")


def main() -> None:
    try:
        df = load_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    sns.set_theme(style="whitegrid")
    print_overview(df)

    print("\n" + "=" * 60)
    print("SAVING PLOTS")
    print("=" * 60)
    plot_loan_distribution(df)
    plot_income_distribution(df)
    plot_loan_amount_distribution(df)
    plot_category_vs_status(df, "Credit_History", "credit_history_vs_approval.png")
    plot_category_vs_status(df, "Education", "education_vs_approval.png")
    plot_correlation_heatmap(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
