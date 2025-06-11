import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Add this
import gdown

# === Setup Paths ===
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
model_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# === Google Drive model download ===
drive_file_id = "1AYvcoSixmaC5rVFGVAh0nWe-S84hqJln"  # replace if different
model_url = f"https://drive.google.com/uc?id={drive_file_id}"

# Make sure model exists, else download
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    st.info("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# === Load model ===
model = tf.keras.models.load_model(model_path)

# === Load class names ===
class_indices = json.load(open(class_indices_path))


# === Image Preprocessing ===
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# === Predict Function ===
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# === Streamlit UI ===
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: **{str(prediction)}**')
