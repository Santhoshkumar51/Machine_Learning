import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# Custom CSS (your changes applied)
# -------------------------
st.markdown("""
<style>
/* App background gradient */
.stApp {
    background-image: linear-gradient(135deg, #60f0df, #beeec5, #e2e8f0);
    min-height: 100vh;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Global readable text */
html, body, [class*="css"] {
    color: #111827 !important;
}

/* Header card */
.header-card {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    padding: 30px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    margin-bottom: 30px;
}

/* Title */
.title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    opacity: 0.9;
}

/* Section header */
.section-header {
    background: linear-gradient(135deg, #e0e7ff, #f5d0fe);
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 700;
    color: #111827 !important;   
    margin-bottom: 16px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}

/* Content cards */
.card {
    background: rgba(255, 255, 255, 0.92);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

/* Metric highlight */
.metric {
    font-size: 28px;
    font-weight: 700;
    color: #4f46e5;
}

/* Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
}

/* Prediction box highlight */
.prediction-box {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    color: #1e40af;
    margin-top: 15px;
}

/* Number input styling */
div[data-testid="stNumberInput"]{
    margin-bottom: 20px;
}

div[data-testid="stNumberInput"] > label {
    font-weight: 600;
    color: #1e3a8a;
}

div[data-testid="stNumberInput"] input {
    background-color: #ad8ef1;
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 1.1rem;
    color: #111827;
    transition: all 0.25s ease;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="header-card">
    <div class="title">Telco Customer Churn</div>
    <div class="subtitle">Logistic Regression • Streamlit App</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Target
df['Churn'] = df['Churn'].astype(str).str.strip()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['Churn'])
df['Churn'] = df['Churn'].astype(int)

# Features
df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']].astype(float)
y = df['Churn']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, Y_train)

# Predictions
Y_pred = model.predict(x_test)
accuracy = accuracy_score(Y_test, Y_pred)

# -------------------------
# Metrics card
# -------------------------
st.markdown(f"""<div class="card">
    <h2 class="section-header">Model Accuracy</h2>
    <div class="metric">{accuracy:.2%}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Confusion Matrix
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred, ax=ax)
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Prediction Section
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Make Your Own Prediction</h2>', unsafe_allow_html=True)

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

if st.button("Predict Churn"):
    input_data = np.array([[tenure, monthly, total]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown('<div class="prediction-box">❌ Customer is likely to CHURN</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box">✅ Customer is NOT likely to churn</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
