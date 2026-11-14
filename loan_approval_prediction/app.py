"""
app.py - Streamlit dashboard for the Loan Approval Prediction System.

Run:  streamlit run app.py
(Train the model first with: python train.py)
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# Same column names and order used in train.py
FEATURE_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Credit_History", "Property_Area",
]

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .main-title {font-size: 2.2rem; font-weight: 700; color: #1f3b63; margin-bottom: 0;}
    .subtitle {color: #5b6b7f; margin-bottom: 1.2rem;}
    .result-approved {background: #e8f6ee; border-left: 6px solid #2e8b57;
                      padding: 1rem 1.2rem; border-radius: 8px; color: #1b5e3a;}
    .result-rejected {background: #fdecea; border-left: 6px solid #c0392b;
                      padding: 1rem 1.2rem; border-radius: 8px; color: #8e2a20;}
    .disclaimer {background: #fff8e1; border-left: 6px solid #f0ad00;
                 padding: 0.8rem 1.2rem; border-radius: 8px; color: #6b5200; margin-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the saved pipeline (preprocessing + classifier) once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found at models/best_model.pkl. Run 'python train.py' first."
        )
    return joblib.load(MODEL_PATH)


def build_input_row(values: dict) -> pd.DataFrame:
    """Create a one-row DataFrame with the exact columns/order used in training."""
    return pd.DataFrame([values])[FEATURE_COLUMNS]


# ---------- Header ----------
st.markdown('<p class="main-title">🏦 Loan Approval Prediction System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">An educational machine learning app that estimates whether a loan '
    "application is likely to be approved or rejected, based on applicant details.</p>",
    unsafe_allow_html=True,
)

# ---------- Load model ----------
try:
    model = load_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # corrupted file, version mismatch, etc.
    st.error(f"Could not load the model: {err}")
    st.info("Re-run 'python train.py' in this environment to rebuild the model.")
    st.stop()

# Make sure the saved model expects the same features as this form
expected = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
if sorted(expected) != sorted(FEATURE_COLUMNS):
    st.error("The saved model was trained on different features. Please re-run 'python train.py'.")
    st.stop()

# ---------- Input form ----------
with st.form("loan_form"):
    st.subheader("Applicant details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital status (Married)", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with col2:
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant income (monthly)", min_value=0,
                                           max_value=1_000_000, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant income (monthly)", min_value=0,
                                             max_value=1_000_000, value=0, step=500)
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    with col3:
        loan_amount = st.number_input("Loan amount (in thousands)", min_value=1,
                                      max_value=1000, value=130, step=5)
        loan_term = st.selectbox("Loan term (months)",
                                 [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
        credit_history = st.selectbox(
            "Credit history",
            options=[1.0, 0.0],
            format_func=lambda v: "Good (meets guidelines)" if v == 1.0 else "Poor (does not meet guidelines)",
        )

    submitted = st.form_submit_button("🔍 Predict Loan Status", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    try:
        row = build_input_row({
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area,
        })

        prediction = int(model.predict(row)[0])  # 1 = Approved (Y), 0 = Rejected (N)
        approval_prob = None
        if hasattr(model, "predict_proba"):
            approval_prob = float(model.predict_proba(row)[0][list(model.classes_).index(1)])

        st.subheader("Prediction result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Loan Status", "Approved (Y)" if prediction == 1 else "Rejected (N)")
        if approval_prob is not None:
            m2.metric("Approval probability", f"{approval_prob * 100:.1f}%")
            m3.metric("Rejection probability", f"{(1 - approval_prob) * 100:.1f}%")
            st.progress(min(max(approval_prob, 0.0), 1.0), text="Approval likelihood")
        else:
            m2.metric("Approval probability", "Not available")

        if prediction == 1:
            st.markdown(
                '<div class="result-approved"><b>✅ Likely to be approved.</b> '
                "Based on the details entered, this application resembles past approved loans.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-rejected"><b>❌ Likely to be rejected.</b> '
                "Based on the details entered, this application resembles past rejected loans.</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="disclaimer"><b>Important:</b> This is a machine-learning estimate for '
            "educational purposes only. It is <b>not</b> a real bank decision and must not be used "
            "to make financial decisions.</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Show the input sent to the model"):
            st.dataframe(row, use_container_width=True)

    except Exception as err:
        st.error(f"Prediction failed: {err}")
else:
    st.info("Fill in the applicant details above and click **Predict Loan Status**.")
