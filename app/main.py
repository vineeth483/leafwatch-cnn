import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

from keras.layers import InputLayer
from keras.mixed_precision import Policy as DTypePolicy

# ─── CONFIG ────────────────────────────────────────────────────────────────
FILE_ID    = "1AYvcoSixmaC5rVFGVAh0nWe-S84hqJln"
MODEL_URL  = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_FILE = "plant_model.h5"
CLASS_FILE = "class_indices.json"
IMG_SIZE   = (224, 224)

# ─── FIX: Custom deserialization for legacy config ──────────────────────────
def patched_input_layer(**config):
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return InputLayer(**config)

CUSTOM_OBJECTS = {
    "InputLayer": patched_input_layer,
    "DTypePolicy": DTypePolicy,
}

# ─── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model…")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        return tf.keras.models.load_model(MODEL_FILE, compile=False)

@st.cache_data
def load_class_map():
    with open(CLASS_FILE, "r") as f:
        return json.load(f)

# ─── PREDICTION HELPERS ─────────────────────────────────────────────────────
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def predict(img: Image.Image):
    x = preprocess(img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    return class_map[str(idx)], float(preds[idx])

# ─── STREAMLIT UI ───────────────────────────────────────────────────────────
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image, then click **Classify**.")

model = load_model()
class_map = load_class_map()

uploaded = st.file_uploader("Choose an image…", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Running inference…"):
            label, conf = predict(img)
        st.success(f"**Prediction:** {label}  \n**Confidence:** {conf:.1%}")
