import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tensorflow as tf

# URL model dari Hugging Face
MODEL_URL = "https://huggingface.co/zakialfadilah/resnet_lite/resolve/main/model_resnet50.tflite"
MODEL_PATH = "model_resnet50.tflite"

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
        with st.spinner("Mengunduh model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Fungsi prediksi gambar
def predict_image(interpreter, image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array.astype(np.float32), axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# Streamlit UI
st.title("Klasifikasi Jajanan Tradisional")

uploaded_file = st.file_uploader("Upload gambar jajanan tradisional", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Load model
    interpreter = download_and_load_model()

    # Prediksi
    prediction = predict_image(interpreter, image)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.subheader("Hasil Prediksi:")
    st.write(f"Label: **{predicted_label}**")
    st.write(f"Akurasi: **{confidence:.2f}%**")

    # Detail probabilitas semua kelas
    st.subheader("Detail Probabilitas:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[i]*100:.2f}%")
