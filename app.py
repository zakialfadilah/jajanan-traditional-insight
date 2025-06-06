import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# URL ke model ResNet50 (.keras)
MODEL_URL = "https://huggingface.co/zakialfadilah/best_model_resnet50/resolve/main/best_model_resnet50.keras"
MODEL_PATH = "best_model_resnet50.keras"

# Daftar kelas
CLASS_NAMES = [
    "kembang_goyang", "kerak_telor", "kue_cente", "kue_cubit", "kue_cucur",
    "kue_gemblong", "kue_lumpur", "kue_pancong", "kue_rangi", "kue_wajik",
    "ongol_ongol", "putu_mayang", "selendang_mayang", "uli_bakar"
]

# Informasi tambahan untuk beberapa kelas
EXTRA_INFO = {
    "kembang_goyang": {
        "history": """
**Kembang Goyang - Historical Insight**  
Kembang Goyang, a beloved traditional snack from Betawi, has roots reaching beyond Indonesian shores. Often compared to Norway’s krumkake—a Viking-era dessert—this flower-shaped rice flour snack is fried with a shaking motion (hence the name "goyang").  
It is believed to have been introduced via Portuguese influence in the 19th century and gradually adapted by locals. Now a staple in Betawi celebrations, Kembang Goyang reflects cultural fusion and historical journeys through culinary evolution.
""",
        "ingredients": """
**Ingredients:**  
**Dry Mix (A):**
- 170g all-purpose flour  
- 100g tapioca flour  
- 50g rice flour  

**Wet Mix (B):**
- 120g sugar  
- 65ml instant coconut milk  
- 2 eggs  
- 1/2 tsp salt  
- A dash of vanilla  
"""
    },
    "kerak_telor": {
        "history": """
**Kerak Telor - Historical Insight**  
Originating in the 1920s from the Betawi community in Menteng, Jakarta, Kerak Telor was originally a luxurious dish enjoyed by the upper class during the Dutch colonial period.  
As time passed, it became accessible to all. By the 1970s, it was being sold around Monas and promoted by Jakarta's governor Ali Sadikin.  
Symbolically, the egg represents leadership, while the spices signify the diversity of society—showing how leaders unify differences into one flavor.
""",
        "ingredients": """
**Ingredients:**  
- 2–3 tbsp cooked rice  
- 1 egg  
- A pinch of salt, pepper, and seasoning  
- Ebi (dried shrimp floss) to taste  
- Fried shallots  
- 1 tbsp spiced coconut (serundeng)  
"""
    }
}

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
    model = load_model(MODEL_PATH)
    return model

def predict_image(model, image):
    image = image.resize((256, 256))  # Sesuai dengan training
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)  # PENTING: Sesuai ResNet50
    prediction = model.predict(image_array)[0]
    return prediction

# ========================
# UI Streamlit Start Here
# ========================
st.title("🍢 Indonesian Traditional Snack Insight")

uploaded_file = st.file_uploader("Upload a traditional snack image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = download_and_load_model()
    prediction = predict_image(model, image)

    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Display prediction result prominently
    st.markdown("## 🧠 Prediction Result")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{predicted_label.replace('_', ' ').title()}</h1>", unsafe_allow_html=True)

    # Show extra info if available
    if predicted_label in EXTRA_INFO:
        st.markdown("---")
        st.markdown(EXTRA_INFO[predicted_label]["history"])
        st.markdown(EXTRA_INFO[predicted_label]["ingredients"])
