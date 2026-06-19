import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Risk Prediction", page_icon="📊")

# ---------------- LOAD CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 class='title'>Customer Risk Prediction System (KNN)</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This system predicts customer risk by comparing them with similar customers.</p>",
    unsafe_allow_html=True
)

# ---------------- DUMMY DATASET ----------------
data = pd.read_csv('credit_risk_dataset.csv')

X = data[["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]]
y = data["loan_status"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input(
    "Annual Income",
    min_value=10000,
    max_value=200000,
    value=50000,
    step=1000   # 🔥 increment by 1000
)

loan_amt = st.sidebar.number_input(
    "Loan Amount",
    min_value=5000,
    max_value=100000,
    value=20000,
    step=1000   # 🔥 increment by 1000
)

credit = st.sidebar.radio("Credit History", ["Yes", "No"])
k_value = st.sidebar.slider("K Value", 1, 15, 5)

credit_val = 1 if credit == "Yes" else 0

user_data = np.array([[age, income, loan_amt, credit_val]])
user_scaled = scaler.transform(user_data)

# ---------------- MODEL ----------------
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_scaled, y)

# ---------------- PREDICTION BUTTON ----------------
if st.button("Predict Customer Risk"):

    prediction = model.predict(user_scaled)[0]
    distances, indices = model.kneighbors(user_scaled)

    neighbor_classes = y.iloc[indices[0]].values
    majority_class = np.bincount(neighbor_classes).argmax()

    # ---------------- RESULT DISPLAY ----------------
    if prediction == 1:
        st.markdown("<div class='high-risk'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='low-risk'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    # ---------------- NEIGHBOR EXPLANATION ----------------
    st.markdown(
        f"""
        <div class='info-box'>
        <b>Neighbors Considered (K):</b> {k_value} <br>
        <b>Majority Class Among Neighbors:</b> {"High Risk" if majority_class == 1 else "Low Risk"}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Optional Table
    st.subheader("Nearest Customers")
    st.dataframe(data.iloc[indices[0]])

    # ---------------- BUSINESS INSIGHT ----------------
    st.markdown(
        "<div class='explanation'>This decision is based on similarity with nearby customers in feature space.</div>",
        unsafe_allow_html=True
    )
