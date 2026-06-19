import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide"
)
MODEL_PATH = "road_damage_cnn_model.h5"

@st.cache_resource

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = ["Pothole", "Crack", "Manhole"]

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------
# HEADER SECTION
# -----------------------------
st.title("🚧 AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

st.markdown("---")

# -----------------------------
# ABOUT PROJECT SECTION
# -----------------------------
st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### Road Monitoring Importance
        Regular road inspection helps reduce accidents,
        improve transportation safety, and support smart city maintenance.
        """
    )

with col2:
    st.success(
        """
        ### Role of CNN
        Convolutional Neural Networks (CNNs) automatically detect
        potholes, cracks, and road defects from images.
        """
    )

with col3:
    st.warning(
        """
        ### Industry Applications
        Used in highway monitoring, smart cities,
        municipal maintenance, and infrastructure analytics.
        """
    )

st.markdown("---")

# -----------------------------
# UPLOAD SECTION
# -----------------------------
st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("---")

    # IMAGE PREVIEW
    st.header("🖼 Uploaded Image Preview")

    st.image(image, caption="Uploaded Road Image", use_container_width=True)

    # PREPROCESS
    processed_image = preprocess_image(image)

    # MODEL PREDICTION
    predictions = model.predict(processed_image)

    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions) * 100)

    predicted_class = class_names[predicted_index]

    # SEVERITY LOGIC
    if confidence >= 85:
        severity = "High"
    elif confidence >= 60:
        severity = "Medium"
    else:
        severity = "Low"

    st.markdown("---")

    # -----------------------------
    # PREDICTION AREA
    # -----------------------------
    st.header("🔍 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", predicted_class)

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    with col3:
        st.metric("Severity", severity)

    st.markdown("---")

    # -----------------------------
    # VISUALIZATION AREA
    # -----------------------------
    st.header("📊 Visualization Area")

    probabilities = predictions[0] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(class_names, probabilities)

    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Damage Classes")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # RECOMMENDATION SECTION
    # -----------------------------
    st.header("⚠ Recommendations")

    if predicted_class == "Pothole":
        st.error(
            "Immediate maintenance recommended. High-risk road condition detected."
        )

    elif predicted_class == "Crack":
        st.warning(
            "Schedule repair soon to prevent further structural damage."
        )

    else:
        st.info(
            "Road condition appears manageable. Continue regular monitoring."
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-Based Smart Road Monitoring System using CNN and Streamlit")