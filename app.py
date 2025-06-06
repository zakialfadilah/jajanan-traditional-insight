import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import plotly.express as px
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Indonesian Traditional Snack Classifier",
    page_icon="🍢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-transform: capitalize;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .ingredients-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
Kembang Goyang, a beloved traditional snack from Betawi, has roots reaching beyond Indonesian shores. Often compared to Norway's krumkake—a Viking-era dessert—this flower-shaped rice flour snack is fried with a shaking motion (hence the name "goyang").  
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
        with st.spinner("🔄 Downloading AI model... Please wait"):
            try:
                response = requests.get(MODEL_URL)
                response.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading model: {str(e)}")
                return None
    
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_image(model, image):
    try:
        image = image.resize((256, 256))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        prediction = model.predict(image_array)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

def create_prediction_chart(prediction, class_names):
    # Get top 5 predictions
    top_indices = np.argsort(prediction)[-5:][::-1]
    top_classes = [class_names[i].replace('_', ' ').title() for i in top_indices]
    top_scores = [prediction[i] * 100 for i in top_indices]
    
    df = pd.DataFrame({
        'Snack': top_classes,
        'Confidence (%)': top_scores
    })
    
    fig = px.bar(df, x='Confidence (%)', y='Snack', orientation='h',
                 title='Top 5 Predictions',
                 color='Confidence (%)',
                 color_continuous_scale='viridis')
    fig.update_layout(height=400, showlegend=False)
    return fig

# ========================
# Main App Interface
# ========================

# Header
st.markdown('<h1 class="main-header">🍢 Indonesian Traditional Snack Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🤖 AI-Powered Recognition of Traditional Indonesian Snacks</p>', unsafe_allow_html=True)

# Sidebar for app info
with st.sidebar:
    st.markdown("## 📱 About This App")
    st.markdown("""
    This AI application uses deep learning to classify traditional Indonesian snacks from images.
    
    **Features:**
    - 🎯 Recognizes 14 different traditional snacks
    - 📊 Shows confidence scores
    - 📚 Provides historical insights
    - 🍳 Includes recipe ingredients
    """)
    
    st.markdown("## 🎯 Supported Snacks")
    for idx, snack in enumerate(CLASS_NAMES, 1):
        st.markdown(f"{idx}. {snack.replace('_', ' ').title()}")
    
    st.markdown("---")
    st.markdown("## 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit images
    - Center the snack in the frame
    - Avoid cluttered backgrounds
    - Use images with good resolution
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload Your Snack Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a traditional Indonesian snack"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        with st.spinner("🔍 Analyzing your image..."):
            model = download_and_load_model()
            
            if model is not None:
                prediction = predict_image(model, image)
                
                if prediction is not None:
                    predicted_index = np.argmax(prediction)
                    predicted_label = CLASS_NAMES[predicted_index]
                    confidence = prediction[predicted_index] * 100
                    
                    # Main prediction result
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div class="prediction-title">{predicted_label.replace('_', ' ')}</div>
                            <div class="confidence-score">Confidence: {confidence:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Prediction chart
                    if st.checkbox("📊 Show detailed predictions", value=True):
                        fig = create_prediction_chart(prediction, CLASS_NAMES)
                        st.plotly_chart(fig, use_container_width=True)

# Additional Information Section
if uploaded_file and 'predicted_label' in locals():
    st.markdown("---")
    
    if predicted_label in EXTRA_INFO:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>📚 Historical Background</h3>
                    {EXTRA_INFO[predicted_label]["history"]}
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="ingredients-card">
                    <h3>🍳 Recipe Ingredients</h3>
                    {EXTRA_INFO[predicted_label]["ingredients"]}
                """, 
                unsafe_allow_html=True
            )
    else:
        st.info("📝 Historical information and ingredients for this snack will be added soon!")

# Welcome message for first-time users
if not uploaded_file:
    st.markdown("---")
    st.markdown("## 🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Upload Image
        Click the upload button and select a clear image of a traditional Indonesian snack.
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ AI Analysis
        Our deep learning model will analyze your image and identify the snack type.
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Learn More
        Discover the history and ingredients of your identified snack!
        """)
    
    st.markdown("""
    ---
    **Ready to explore Indonesian traditional snacks?** 
    Upload an image above to begin your culinary journey! 🌟
    """)
