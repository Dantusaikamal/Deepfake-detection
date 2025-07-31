# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 128
model = load_model("deepfake_model.h5")

# Page Config
st.set_page_config(
    page_title="Deepfake Detector",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background: linear-gradient(to bottom right, #ffffff, #e6f0ff);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 0 25px rgba(0,0,0,0.05);
    }
    h1 {
        color: #1f4e79;
    }
    .stButton>button {
        background: linear-gradient(to right, #0073e6, #0059b3);
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff6a00 , #ee0979);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("## Deepfake Detection AI", unsafe_allow_html=True)
st.markdown("Upload a face image below and the model will predict if it's a **real** or **deepfake** face.")

# Upload and Preview
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "⚠️ Deepfake" if prediction < 0.5 else "Real"
    confidence = (1 - prediction) if prediction < 0.5 else prediction

    # Result Card
    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.progress(float(confidence))

    # Confidence Score
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
else:
    st.info("Upload an image to begin detection.")

st.markdown("</div>", unsafe_allow_html=True)
