import os, json, gdown
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit as st

# Import from keras (not tf.keras) to match the saved module path:
from keras.mixed_precision import Policy as KerasPolicy
from keras.layers import InputLayer as KerasInputLayer
import tensorflow as tf

# ─── Constants ──────────────────────────────────────────────────
DRIVE_FILE_ID = "1AYvcoSixmaC5rVFGVAh0nWe-S84hqJln"
MODEL_DIR     = "trained_model"
MODEL_NAME    = "plant_disease_prediction_model.h5"
CLASS_JSON    = "class_indices.json"
IMG_SIZE      = (224, 224)

# ─── Legacy InputLayer shim ─────────────────────────────────────
def patched_input_layer(**config):
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return KerasInputLayer(**config)

# ─── Custom objects mapping ──────────────────────────────────────
CUSTOM_OBJECTS = {
    # Match what your .h5 metadata wrote:
    "InputLayer": patched_input_layer,
    "DTypePolicy": KerasPolicy,
}

@st.cache_resource
def load_model():
    # ensure directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model from Google Drive…")
        gdown.download(url, path, quiet=False)

    # Use custom_object_scope so that Keras recognizes both of our shims
    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        return tf.keras.models.load_model(path, compile=False)

@st.cache_data
def load_class_indices():
    with open(CLASS_JSON, "r") as f:
        return json.load(f)

def preprocess(img_buf: BytesIO) -> np.ndarray:
    img = Image.open(img_buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[None, ...]

def predict(model, classes, buf: BytesIO):
    x = preprocess(buf)
    preds = model.predict(x)[0]
    idx   = int(np.argmax(preds))
    return classes[str(idx)], float(preds[idx])

# ─── Streamlit UI ───────────────────────────────────────────────
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image, then click **Classify**.")

model   = load_model()
classes = load_class_indices()

uploaded = st.file_uploader("Choose an image…", type=["jpg","png","jpeg"])
if uploaded:
    st.image(uploaded, use_column_width=True, caption="Your upload")
    if st.button("Classify"):
        with st.spinner("Running inference…"):
            label, conf = predict(model, classes, uploaded)
        st.success(f"**Prediction:** {label}  \n**Confidence:** {conf:.1%}")
