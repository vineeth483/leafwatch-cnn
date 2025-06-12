import os, json, gdown
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from PIL import Image
import streamlit as st

# ─── Constants ─────────────────────────────────────────────────────────────────
DRIVE_FILE_ID = "1AYvcoSixmaC5rVFGVAh0nWe-S84hqJln"
MODEL_DIR     = "trained_model"
MODEL_NAME    = "plant_disease_prediction_model.h5"
CLASS_JSON    = "class_indices.json"
IMG_SIZE      = (224, 224)

# ─── Custom InputLayer shim ────────────────────────────────────────────────────
def patched_input_layer(**config):
    # if it's coming with legacy "batch_shape", rename it to what the new class expects:
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return InputLayer(**config)


# ─── Download & Load Helpers ──────────────────────────────────────────────────
@st.cache_resource
def load_model_from_drive():
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model from Google Drive…")
        gdown.download(url, path, quiet=False)
    # Here we tell load_model to use our patched InputLayer
    return tf.keras.models.load_model(
        path,
        custom_objects={"InputLayer": patched_input_layer},
        compile=False
    )

@st.cache_data
def load_class_indices():
    with open(CLASS_JSON, "r") as f:
        return json.load(f)


# ─── Preprocessing & Prediction ───────────────────────────────────────────────
def preprocess(image_buf: BytesIO) -> np.ndarray:
    img = Image.open(image_buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(model, classes, buf: BytesIO):
    x = preprocess(buf)
    preds = model.predict(x)[0]
    idx  = int(np.argmax(preds))
    return classes[str(idx)], float(preds[idx])


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image and click **Classify**.")

model = load_model_from_drive()
classes = load_class_indices()

uploaded = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
if uploaded:
    st.image(uploaded, caption="Your upload", use_column_width=True)
    if st.button("Classify"):
        with st.spinner("Running inference…"):
            label, conf = predict(model, classes, uploaded)
        st.success(f"**Prediction:** {label}  \n**Confidence:** {conf:.1%}")
