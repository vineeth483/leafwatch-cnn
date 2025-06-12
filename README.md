
# 🌿 LeafWatch – Plant Disease Detection App

**LeafWatch** is a deep learning-powered web application that detects plant diseases from leaf images using a custom-trained Convolutional Neural Network (CNN). The model is trained in Keras and deployed via a lightweight Streamlit interface.

## 🚀 Live Demo  
👉 [Streamlit App](https://vineeth-483.streamlit.app/)  
📦 [GitHub Repo](https://github.com/vineeth483/leafwatch-cnn)

---

## 🧠 Model Overview
- Built a **4-block CNN** architecture with Conv2D → ReLU → MaxPooling layers and a Dense classification head.
- Trained on a multi-class plant leaf dataset (e.g., Apple Scab, Black Rot, Cedar Rust, Healthy).
- Achieved **~92% validation accuracy** after 20 epochs using data augmentation and Adam optimizer.
- Exported as `.h5` for deployment and mapped class indices using `class_indices.json`.

---

## 📊 Training Highlights
- Image size: **224×224**, normalized pixel values  
- Augmentation: Flips, zoom, rotation via `ImageDataGenerator`  
- Frameworks used: **TensorFlow/Keras**, **Matplotlib** for accuracy/loss plots  
- Output:  
  - `plant_model.h5` – Trained model weights  
  - `class_indices.json` – Mapping of class labels  

---

## 🖥️ Streamlit App Features
- Upload any plant leaf image (JPG, PNG).  
- The app **auto-downloads** the model via `gdown` from Google Drive.  
- Displays the predicted **disease class** and **confidence score**.  
- Deployed on **Streamlit Cloud** with minimal startup delay.

---

## 📁 Folder Structure
```
├── app/
│   ├── main.py               ← Streamlit app
│   ├── requirements.txt      ← Dependencies
│   ├── class_indices.json    ← Class label mapping
│   └── runtime.txt           ← Python version pin
├── model_training_notebook/
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
```

---

## ⚙️ How to Run Locally (Optional)
```bash
git clone https://github.com/vineeth483/leafwatch-cnn.git
cd app
pip install -r requirements.txt
streamlit run main.py
```

---

## 📌 Technologies Used
- Python, TensorFlow/Keras  
- Streamlit, gdown  
- Matplotlib, Pillow, NumPy  

---

## 🧑‍💻 Author
Made with 🌱 by [Vineeth Bammidi](https://github.com/vineeth483)
