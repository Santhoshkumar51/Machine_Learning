import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page Config
st.set_page_config(page_title="Multiple Linear Regression", layout="centered")

# Load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles1.css")

# Header
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> using <b>Total Bill</b> and <b>Party Size</b></p>
</div>
""", unsafe_allow_html=True)

# Dataset Preview
@st.cache_data
def load_dataset():
    return sns.load_dataset("tips")

df = load_dataset()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill", "size", "tip"]].head())
st.markdown("</div>", unsafe_allow_html=True)

# Feature & Target
X = df[["total_bill", "size"]]
y = df["tip"]

# Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

n = len(y_test)
k = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

#Visualizations#
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h1>Model Visualization</h1>', unsafe_allow_html=True)
plt.title("Total Bill vs Tip Amount with Multiple Linear Regression Line")
plt.xlabel("Total Bill and Size")
plt.ylabel("Tip Amount")
plt.scatter(df["total_bill"], y, color='blue', label='Actual Tips')
plt.plot(df["total_bill"], model.predict(scaler.transform(X)), color='red', linewidth=2, label='Predicted Tips')
plt.legend()
st.pyplot(plt.gcf())
st.markdown('</div>', unsafe_allow_html=True)

# Model Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.2f}")
c4.metric("RMSE", f"{rmse:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# Coefficients
st.markdown(f"""
<div class="card">
    <h3>Model Parameters</h3>
    <p>
        <b>Coefficient (Total Bill):</b> {model.coef_[0]:.3f}<br>
        <b>Coefficient (Size):</b> {model.coef_[1]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($):",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)

size = st.slider(
    "Party Size:",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)

input_scaled = scaler.transform([[bill, size]])
predicted_tip = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${predicted_tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
