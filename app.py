import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model

# URL model dari Hugging Face Hub
# MODEL_URL = "https://huggingface.co/zakialfadilah/best_model_resnet50/resolve/main/best_model_resnet50.keras"
MODEL_URL = "https://huggingface.co/zakialfadilah/model_vgg/resolve/main/model_vgg16.tflite"
MODEL_PATH = "model_vgg16.tflite"
# MODEL_PATH = "best_model_resnet50.keras"

# Daftar kelas (label)
CLASS_NAMES = [
    "kembang_goyang", "kerak_telor", "kue_cente", "kue_cubit", "kue_cucur",
    "kue_gemblong", "kue_lumpur", "kue_pancong", "kue_rangi", "kue_wajik",
    "ongol_ongol", "putu_mayang", "selendang_mayang", "uli_bakar"
]

# Download dan load model hanya sekali
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
    model = load_model(MODEL_PATH)
    return model

# Fungsi prediksi gambar
def predict_image(model, image):
    image = image.resize((256, 256))  # Sesuaikan dengan input model
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0]  # Ambil hasil prediksi dari batch
    return prediction

# Streamlit UI
st.title("Klasifikasi Jajanan Tradisional")

uploaded_file = st.file_uploader("Upload gambar jajanan tradisional", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Load model
    model = download_and_load_model()

    # Prediksi
    prediction = predict_image(model, image)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.subheader("Hasil Prediksi:")
    st.write(f"Label: **{predicted_label}**")
    st.write(f"Akurasi: **{confidence:.2f}%**")

    # Optional: tampilkan semua probabilitas
    st.subheader("Detail Probabilitas:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[i]*100:.2f}%")
