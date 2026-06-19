import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Loan Approval System Using SVM",
    page_icon="💳",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 class='title'>Smart Loan Approval System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This system uses <b>Support Vector Machines (SVM)</b> "
    "to predict loan approval using real banking data.</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# ---------------- DATA CLEANING ----------------
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Encode target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Encode categorical columns
encoder = LabelEncoder()
for col in ['Self_Employed', 'Property_Area']:
    df[col] = encoder.fit_transform(df[col])

# ---------------- FEATURES & TARGET ----------------
X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History',
        'Self_Employed', 'Property_Area']]
y = df['Loan_Status']

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, step=500)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0, step=10)
credit = st.sidebar.radio("Credit History", ["Yes", "No"])
self_emp = st.sidebar.radio("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# Encode inputs
credit_val = 1 if credit == "Yes" else 0
self_emp_val = 1 if self_emp == "Yes" else 0
property_val = encoder.fit_transform(
    df['Property_Area'].astype(str)
)[['Urban', 'Semiurban', 'Rural'].index(property_area)]

X_input = np.array([[income, loan_amt, credit_val,
                     self_emp_val, property_val]])
X_input_scaled = scaler.transform(X_input)

# ---------------- MODEL SELECTION ----------------
st.sidebar.header("⚙️ SVM Kernel")

kernel_choice = st.sidebar.radio(
    "Select Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel_choice == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
if st.button("Check Loan Eligibility"):
    prediction = model.predict(X_input_scaled)[0]
    confidence = model.predict_proba(X_input_scaled)[0].max() * 100

    if prediction == 1:
        st.markdown("<div class='approved'>✅ Loan Approved</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='rejected'>❌ Loan Rejected</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='info-box'>
        <b>Kernel Used:</b> {kernel_choice}<br>
        <b>Confidence Score:</b> {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    explanation = (
        "Based on strong credit history and income pattern, "
        "the applicant is likely to repay the loan."
        if prediction == 1 else
        "Based on weak credit history or income pattern, "
        "the applicant is unlikely to repay the loan."
    )

    st.markdown(f"<div class='explanation'>{explanation}</div>", unsafe_allow_html=True)
