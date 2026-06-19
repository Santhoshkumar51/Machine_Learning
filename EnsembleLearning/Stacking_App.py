import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart House Price Prediction", page_icon="🏠")

# ---------------- LOAD CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 class='title'>🏠 Smart House Price Prediction – Stacking Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This system predicts house prices using a Stacking Ensemble model that combines multiple ML algorithms for better accuracy.</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("kc_house_data.csv")

# Select useful features
features = ['bedrooms','bathrooms','sqft_living','floors','grade','sqft_above','lat','long']
X = df[features]
y = df['price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------- STACKING MODEL ----------------
base_models = [
    ('lr', LinearRegression()),
    ('dt', DecisionTreeRegressor()),
    ('knn', KNeighborsRegressor())
]

regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)

regressor.fit(X_train, y_train)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🏠 House Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 2.0)
sqft = st.sidebar.number_input("Living Area (sqft)", step=100)
floors = st.sidebar.slider("Floors", 1.0, 3.0, 1.0)
grade = st.sidebar.slider("Grade", 1, 13, 7)
sqft_above = st.sidebar.number_input("Sqft Above Ground", step=100)
lat = st.sidebar.number_input("Latitude", format="%.4f")
long = st.sidebar.number_input("Longitude", format="%.4f")

user_data = np.array([[bedrooms,bathrooms,sqft,floors,grade,sqft_above,lat,long]])
user_scaled = scaler.transform(user_data)

# ---------------- MODEL ARCHITECTURE DISPLAY ----------------
st.subheader("🧠 Stacking Model Architecture")
st.markdown("""
**Base Models Used**
- Linear Regression  
- Decision Tree Regressor  
- KNN Regressor  

**Meta Model**
- Linear Regression
""")

# ---------------- PREDICTION BUTTON ----------------
if st.button("💰 Predict House Price (Stacking Model)"):

    pred_lr = regressor.named_estimators_['lr'].predict(user_scaled)[0]
    pred_dt = regressor.named_estimators_['dt'].predict(user_scaled)[0]
    pred_knn = regressor.named_estimators_['knn'].predict(user_scaled)[0]

    final_price = regressor.predict(user_scaled)[0]

    st.markdown(f"<div class='price-box'>🏷️ Estimated Price: ${abs(final_price):,.0f}</div>", unsafe_allow_html=True)

    st.subheader("📊 Base Model Predictions")
    st.write(f"Linear Regression → ${pred_lr:,.0f}")
    st.write(f"Decision Tree → ${pred_dt:,.0f}")
    st.write(f"KNN Regressor → ${pred_knn:,.0f}")

    st.subheader("🧠 Final Stacking Decision")
    st.write("The meta-model combines predictions from all base models to produce the final price.")

    st.markdown("""
    <div class='explanation'>
    This prediction is based on property features like size, grade, and location.  
    The stacking model improves accuracy by learning how to best combine multiple model predictions.
    </div>
    """, unsafe_allow_html=True)
