import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error

# Page Config
st.set_page_config(page_title="Linear Regression", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

st.markdown("""
<div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression</p>
</div>
""", unsafe_allow_html=True)

#Dataset Preview
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

#Prepare Data

x=df[['total_bill']]
y=df['tip']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

#metrics
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
n = len(y_test)
k = x.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

#Visualization

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(x,y,alpha=0.6)
ax.plot(x,model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total bill $")
ax.set_ylabel("Tip")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#Model Performance

st.markdown('<div id="metric-container">',unsafe_allow_html=True)
st.subheader('Model Performance')
c1,c2=st.columns(2)
c1.metric("MAE: ",f"{mae:.2f}")
c2.metric("MSE: ",f"{mse:.2f}")
c3,c4=st.columns(2)
c3.metric("R2: ",f"{r2:.2f}")
c4.metric("RMSE: ",f"{rmse:.2f}")
st.markdown('</div>',unsafe_allow_html=True)

#Model Coefficient and Intercept

st.markdown(f"""
<div class="card">
    <h3>Model Interception</h1>
    <p><b> Coefficient(Slope): </b>{model.coef_[0]:.3f}<br>
    <b> Intercept: </b>{model.intercept_:.3f}</p>
</div>
""", unsafe_allow_html=True)

#prediction
# st.subheader("Make Your Own Predictions")
# total_bill_input = st.number_input("Enter Total Bill Amount:", min_value=0.0, step=0.5)
# if st.button("Predict Tip"):
#     scaled_input = scaler.transform([[total_bill_input]])
#     predicted_tip = model.predict(scaled_input)[0]
#     st.markdown(f'<div class="prediction-box">Predicted Tip Amount: {predicted_tip:.2f}</div>', unsafe_allow_html=True)
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Predicted Tip Amount")
bill=st.slider("Total Bill ($):",float(df['total_bill'].min()),float(df["total_bill"].max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box"> Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)