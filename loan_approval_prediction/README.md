# Loan Approval Prediction System

An educational machine learning project that predicts whether a loan application is likely to be **approved** or **rejected** from applicant details. It is **not** a real banking system.

## Features
- Exploratory data analysis with saved plots (`eda.py`)
- Three models compared: Logistic Regression, Decision Tree, Random Forest (`train.py`)
- Leak-free preprocessing with `ColumnTransformer` + `Pipeline`
- Best model saved with Joblib and served in a Streamlit dashboard (`app.py`)

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Joblib.

## Dataset
The project uses the standard **Loan Prediction** dataset (commonly distributed on Kaggle as "Loan Prediction Problem Dataset" / Analytics Vidhya). Required columns:

`Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`

Save the CSV as **`data/loan_prediction.csv`**. No data is generated or substituted automatically; the scripts stop with a clear message if the file is missing.

## Setup (Windows PowerShell)

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (if blocked, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt

# 4. Place the dataset
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item "C:\path\to\your\Loan_Data.csv" "data\loan_prediction.csv"

# 5. Run EDA (plots go to outputs\)
python eda.py

# 6. Train and compare models (saves models\best_model.pkl)
python train.py

# 7. Launch the Streamlit app
streamlit run app.py
```

## Project structure
```
loan_approval_prediction/
├── app.py              # Streamlit dashboard
├── train.py            # Training, comparison, model saving
├── eda.py              # Exploratory data analysis
├── requirements.txt
├── README.md
├── data/
│   └── loan_prediction.csv     # you provide this
├── models/
│   └── best_model.pkl          # created by train.py
└── outputs/                    # plots created by eda.py and train.py
    ├── loan_distribution.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── (income, loan amount, credit history and education plots)
```

## Model evaluation
- **Split:** 80% train / 20% test, `random_state=42`, stratified on `Loan_Status`.
- **Preprocessing (inside the Pipeline, fitted on training data only):** median imputation + `StandardScaler` for numeric columns; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical columns (including `Credit_History`, a 0/1 flag). Unseen categories are ignored safely.
- **Metrics:** accuracy, precision, recall, F1-score, confusion matrix and classification report are printed for each model; a comparison table and `outputs/model_comparison.png` are produced.
- **Model selection:** the model with the highest **5-fold cross-validated F1-score on the training set** is saved (test-set F1 breaks ties). F1 balances precision and recall, which matters because the classes are imbalanced (about 69% approved), and choosing on training-set CV keeps the test set untouched for a fair final score.
- Note: all models predict "Approved" very reliably but miss some rejections (recall for the Rejected class is around 55-60%). Treat predictions as estimates only.

## Disclaimer
This project is for learning only. Its output is a statistical estimate, not a real bank decision.
