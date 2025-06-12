import os
import json
from io import BytesIO

import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.mixed_precision import Policy as KerasPolicy
from keras.layers import InputLayer as KerasInputLayer

# ─── Constants ─────────────────────────────────────────────────────────────────
DRIVE_FILE_ID = "1AYvcoSixmaC5rVFGVAh0nWe-S84hqJln"
MODEL_DIR     = "trained_model"
MODEL_NAME    = "plant_disease_prediction_model.h5"
CLASS_JSON    = "class_indices.json"
IMG_SIZE      = (224, 224)

# ─── Legacy InputLayer shim ────────────────────────────────────────────────────
def patched_input_layer(**config):
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return KerasInputLayer(**config)

# ─── Custom objects mapping ────────────────────────────────────────────────────
CUSTOM_OBJECTS = {
    "InputLayer": patched_input_layer,
    "DTypePolicy": KerasPolicy,
}

@st.cache_resource
def load_model():
    """Download the model from Drive if missing, then load with custom objects."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model from Google Drive…")
        gdown.download(url, model_path, quiet=False)

    # Ensure our custom objects are in scope
    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_class_indices():
    """Load mapping from class-index to human label."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path  = os.path.join(script_dir, CLASS_JSON)
    with open(json_path, "r") as f:
        return json.load(f)

def preprocess_image(buf: BytesIO) -> np.ndarray:
    """Read buffer, resize, normalize, and add batch dimension."""
    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def predict(model, classes, buf: BytesIO):
    """Run inference and return (label, confidence)."""
    x = preprocess_image(buf)
    preds = model.predict(x)[0]
    idx   = int(np.argmax(preds))
    return classes[str(idx)], float(preds[idx])

# ─── Streamlit App ─────────────────────────────────────────────────────────────
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image, then click **Classify** to see the result.")

model   = load_model()
classes = load_class_indices()

uploaded = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    st.image(uploaded, caption="Your upload", use_container_width=True)
    if st.button("Classify"):
        with st.spinner("Running inference…"):
            label, conf = predict(model, classes, uploaded)
        st.success(f"**Prediction:** {label}  \n**Confidence:** {conf:.1%}")
