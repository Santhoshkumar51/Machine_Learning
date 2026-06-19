import streamlit as st
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
BACKGROUND
========================================================= */

.stApp{

    background:
    linear-gradient(
        135deg,
        #8de08f 0%,
        #77d5e7 45%,
        #a46bd6 100%
    );

    background-attachment: fixed;

    overflow-x:hidden;

    color:white;
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.block-container{

    max-width:1400px;

    padding-top:1rem;

    padding-bottom:2rem;
}

/* =========================================================
HEADER
========================================================= */

.main-title{

    text-align:center;

    font-size:64px;

    font-weight:900;

    color:white;

    margin-bottom:8px;

    letter-spacing:-1px;

    text-shadow:
    0px 6px 20px rgba(0,0,0,0.35);
}

.sub-title{

    text-align:center;

    font-size:22px;

    color:#f1f5f9;

    margin-bottom:35px;

    font-weight:500;
}

/* =========================================================
CARDS
========================================================= */

.card{

    background:rgba(15,23,42,0.74);

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    -webkit-backdrop-filter: blur(18px);

    border-radius:28px;

    padding:28px;

    margin-bottom:18px;

    box-shadow:
    0px 10px 35px rgba(0,0,0,0.35);

    transition:0.35s ease;
}

/* Hover effect */

.card:hover{

    transform:translateY(-5px);

    box-shadow:
    0px 18px 45px rgba(0,0,0,0.45);
}

/* =========================================================
HEADINGS
========================================================= */

.card h1,
.card h2,
.card h3{

    color:#7dd3fc;

    margin-bottom:16px;

    font-weight:800;
}

/* =========================================================
TEXT
========================================================= */

.card p,
.card li{

    color:#f8fafc;

    font-size:16px;

    line-height:1.9;
}

/* =========================================================
UPLOAD BOX
========================================================= */
[data-testid="stFileUploader"]{

    background:
    linear-gradient(
        135deg,
        rgba(15,23,42,0.95),
        rgba(30,58,138,0.88)
    );

    border:2px dashed rgba(96,165,250,0.55);

    border-radius:24px;

    padding:22px;

    transition:0.35s ease;

    box-shadow:
    0px 8px 25px rgba(30,64,175,0.35);
}

/* Hover */

[data-testid="stFileUploader"]:hover{

    border:2px dashed #60a5fa;

    background:
    linear-gradient(
        135deg,
        rgba(15,23,42,1),
        rgba(37,99,235,0.92)
    );

    box-shadow:
    0px 0px 30px rgba(59,130,246,0.65);

    transform:translateY(-2px);
}
/* Upload button */

[data-testid="stFileUploader"] section button{

    background:
    linear-gradient(
        135deg,
        #1d4ed8,
        #2563eb
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:700 !important;

    transition:0.3s ease;
}

/* Upload button hover */

[data-testid="stFileUploader"] section button:hover{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    ) !important;

    transform:scale(1.03);
}
            


/* =========================================================
METRICS
========================================================= */

[data-testid="metric-container"]{

    background:rgba(15,23,42,0.88);

    border:1px solid rgba(255,255,255,0.08);

    padding:18px;

    border-radius:22px;

    box-shadow:
    0px 6px 20px rgba(0,0,0,0.35);

    transition:0.3s;
}

[data-testid="metric-container"]:hover{

    transform:translateY(-4px);
}

/* Metric labels */

[data-testid="metric-container"] label{

    color:#cbd5e1 !important;

    font-size:15px;
}

/* Metric values */

[data-testid="metric-container"] div{

    color:white !important;

    font-size:28px;
}

/* =========================================================
IMAGES
========================================================= */

img{

    border-radius:22px;
}

/* =========================================================
ALERTS
========================================================= */

.stSuccess,
.stWarning,
.stError,
.stInfo{

    border-radius:18px;
}

/* =========================================================
FOOTER
========================================================= */

.footer{

    text-align:center;

    margin-top:35px;

    color:#e2e8f0;

    font-size:15px;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar{

    width:10px;
}

::-webkit-scrollbar-thumb{

    background:#1e3a8a;

    border-radius:20px;
}

/* =========================================================
HIDE STREAMLIT
========================================================= */

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

header{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():

    import os

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "road_damage_model.h5"
    )

    return load_model(MODEL_PATH)

model = load_cnn_model()

# ============================================================
# LOAD LABELS
# ============================================================

with open("label_mapping.json", "r") as f:

    label_mapping = json.load(f)

index_to_label = {
    value:key for key, value in label_mapping.items()
}

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AI-Based Road Damage Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Smart City Infrastructure Monitoring using CNN</div>',
    unsafe_allow_html=True
)

# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns([1.15,1])

# ============================================================
# LEFT SECTION - ABOUT
# ============================================================

with left_col:

    st.markdown("""
    <div class="card">

    <h2>About the Project</h2>

    <p>
    Road monitoring is extremely important for public safety,
    smart transportation systems, and infrastructure management.
    Delayed detection of road damage can increase:
    </p>

    <ul>
        <li>Road accidents</li>
        <li>Vehicle damage</li>
        <li>Traffic congestion</li>
        <li>Maintenance costs</li>
    </ul>

    <h3>Role of CNN in Computer Vision</h3>

    <p>
    Convolutional Neural Networks (CNNs) automatically learn
    road surface patterns such as potholes, cracks,
    and texture features for intelligent classification.
    </p>

    <h3>Industry Applications</h3>

    <ul>
        <li>Smart City Monitoring</li>
        <li>Road Safety Systems</li>
        <li>Municipal Infrastructure Analysis</li>
        <li>Autonomous Vehicle Navigation</li>
        <li>Highway Inspection Automation</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RIGHT SECTION
# upload + prediction
# ============================================================

with right_col:

    st.markdown("""
    <div class="card">

    <h2>Upload Road Image</h2>

    <p>
    Upload a road image for AI-powered road damage analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        IMG_SIZE = 128

        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        label = index_to_label[predicted_class]

        # ====================================================
        # SEVERITY
        # ====================================================

        label_lower = label.lower()

        if "pothole" in label_lower:

            severity = "High"

            recommendation = """
            Immediate maintenance recommended.
            High-risk road condition detected.
            """

        elif "crack" in label_lower:

            severity = "Medium"

            recommendation = """
            Scheduled maintenance recommended.
            Surface deterioration detected.
            """

        else:

            severity = "Low"

            recommendation = """
            Routine inspection recommended.
            Moderate infrastructure issue detected.
            """

        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.markdown("""
        <div class="card">

        <h2>Prediction Results</h2>

        </div>
        """, unsafe_allow_html=True)

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Prediction",
            label
        )

        metric2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        metric3.metric(
            "Severity",
            severity
        )

# ============================================================
# SECOND ROW
# IMAGE + GRAPH
# ============================================================

if uploaded_file is not None:

    st.markdown(
        "<div style='margin-top:5px'></div>",
        unsafe_allow_html=True
    )

    row2_left, row2_right = st.columns([1,1])

    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    with row2_left:

        st.markdown("""
        <div class="card">

        <h2>Uploaded Image Preview</h2>

        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # ========================================================
    # GRAPH
    # ========================================================

    with row2_right:

        st.markdown("""
        <div class="card">

        <h2>Class Confidence Graph</h2>

        </div>
        """, unsafe_allow_html=True)

        class_names = list(index_to_label.values())

        probabilities = prediction[0] * 100

        fig, ax = plt.subplots(figsize=(7,4))

        bars = ax.bar(
            class_names,
            probabilities,
            color=["#38bdf8", "#818cf8", "#22c55e"]
        )

        # Dark graph styling

        ax.set_facecolor("#0f172a")

        fig.patch.set_facecolor("#0f172a")

        ax.tick_params(colors='white')

        ax.yaxis.label.set_color('white')

        ax.xaxis.label.set_color('white')

        ax.title.set_color('white')

        ax.set_ylabel("Confidence (%)")

        ax.set_xlabel("Damage Classes")

        ax.set_title("Prediction Probability Chart")

        st.pyplot(fig)

    # ========================================================
    # FINAL ROW
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(f"""
    <div class="card">

    <h2>Maintenance Recommendations</h2>

    <div style="
        background:rgba(255,255,255,0.06);
        padding:18px;
        border-radius:18px;
        margin-top:10px;
        line-height:1.9;
    ">

    <p>
    <b>Repair Priority:</b> {severity}
    </p>

    <p>
    {recommendation}
    </p>

    </div>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Developed using CNN • Keras • Streamlit</div>',
    unsafe_allow_html=True
)