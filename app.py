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

# Additional information for snack classes
EXTRA_INFO = {
    "kembang_goyang": {
        "history": """
**Kembang Goyang - Historical Insight**  
Kembang Goyang, known as a traditional Betawi snack, has fascinating origins beyond Indonesian shores. This flower-shaped rice flour snack, fried with a shaking motion (hence "goyang"), shares similarities with Norway's krumkake—a Viking-era dessert. 

The snack is believed to have been introduced through Portuguese influence in the 19th century and was gradually adapted by locals. It has now become a staple in Betawi celebrations, reflecting cultural fusion and historical culinary journeys. Today, people enjoy Kembang Goyang in various flavor variations, making it a popular snack for different occasions.
""",
        "ingredients": """
**Ingredients:**  
**Mix A (Dry):**
- 170g all-purpose flour  
- 100g tapioca flour  
- 50g rice flour  

**Mix B (Wet):**
- 120g granulated sugar  
- 65ml instant coconut milk (1 pack)  
- 2 eggs  
- 1/2 tsp salt  
- A dash of vanilla  
"""
    },
    "kerak_telor": {
        "history": """
**Kerak Telor - Historical Insight**  
Created in the 1920s, Kerak Telor originated from the Betawi community in Menteng, Central Jakarta. During the Dutch colonial period, this dish was considered expensive and could only be enjoyed by the upper class.

Over time, it became accessible to all social classes. In the 1970s, it began being sold around the National Monument (Monas) and was promoted by Jakarta's governor Ali Sadikin. The dish carries philosophical meaning: the egg represents leadership, while the various spices symbolize individual differences in society—showing how leaders can unite diversity into one harmonious flavor.
""",
        "ingredients": """
**Ingredients:**  
- 2-3 tbsp cooked rice  
- 1 egg  
- Pinch of salt, pepper, and seasoning  
- Ebi (dried shrimp) to taste  
- Fried shallots  
- 1 tbsp serundeng (spiced coconut)  
"""
    },
    "kue_cente": {
        "history": """
**Kue Cente - Historical Insight**  
Based on its name, this cake is believed to originate from China. In Chinese, "cente" means beautiful, while "manis" means sweet. One version suggests that Kue Cente Manis was brought by Chinese traders who came to the archipelago and was then adapted by the Betawi community.

Another version states that Kue Cente Manis is originally from Betawi, made with ingredients easily found in the region such as hunkwe flour, coconut milk, sugar, and sago pearls. Regardless of its origins, Kue Cente Manis has become part of Betawi culture.
""",
        "ingredients": """
**Ingredients:**  
- 50g hunkwe flour (1 pack)  
- 1/2 tbsp all-purpose flour  
- 90g granulated sugar  
- 300ml coconut milk (from 1 small kara pack)  
- A pinch of salt  
- 50g dried sago pearls  
- 2 glasses of water  
"""
    },
    "kue_cubit": {
        "history": """
**Kue Cubit - Historical Insight**  
Kue Cubit actually originates from the Netherlands! The presence during Dutch colonialism created this cultural acculturation. Many cakes we know today are part of Dutch heritage, such as kastengels, kroket, and lapis legit.

The Tulip country didn't just colonize Indonesia but left many culinary legacies, including Kue Cubit. In Dutch, it's called "poffertjes," which became known as Kue Cubit in Indonesia.
""",
        "ingredients": """
**Ingredients:**  
- 130g all-purpose flour  
- 100g granulated sugar  
- 100g butter, melted  
- 3 eggs  
- 150ml white milk  
- 1/2 tsp baking powder  
- 1/4 tsp baking soda  
- 1/2 tsp vanilla powder  
- 1/4 tsp salt  
"""
    },
    "kue_gemblong": {
        "history": """
**Kue Gemblong - Historical Insight**  
Gemblong is made from glutinous rice flour mixed with grated coconut. The outside is coated with liquid palm sugar that crystallizes when dried. This traditional Javanese snack is commonly found along Puncak road, West Java, and often brought as souvenirs.

In Central Java, East Java, and Yogyakarta, people know gemblong by the name "kue getas." Some also call it "kemplang," though these traditional snacks actually have differences despite similar taste and texture.
""",
        "ingredients": """
**Ingredients:**  
- 150g glutinous rice flour  
- 2 tbsp rice flour  
- 150ml warm coconut milk  
- 100g grated coconut  
- 1/2 tsp salt  

**Palm Sugar Coating:**
- 100g shredded palm sugar  
- 100g palm sugar  
- 100ml water  
- 1 pandan leaf  
"""
    },
    "kue_lumpur": {
        "history": """
**Kue Lumpur - Historical Insight**  
Kue Lumpur is a traditional Indonesian snack made from potatoes, flour, sugar, and eggs, usually decorated with raisins and grated coconut. Many historians state that Kue Lumpur was introduced by the Portuguese during their colonization of Indonesia from 1511 to 1595.

Kue Lumpur was inspired by a Portuguese specialty called "pasteis de nata," usually made from custard (a mixture of milk and egg yolks). When in Indonesia, the Portuguese adapted the recipe using ingredients commonly found in Indonesia, such as potatoes or flour. Kue Lumpur was first made in the Betawi region.
""",
        "ingredients": """
**Ingredients:**  
- 500g potatoes (after steaming)  
- 500ml coconut milk  
- 250g all-purpose flour  
- 250g granulated sugar  
- 2 whole eggs  
- 1 egg yolk  
- 100g butter, melted  
- 1/2 tsp salt  
- 1/2 tsp vanilla  
- Raisins for topping  
"""
    },
    "kue_pancong": {
        "history": """
**Kue Pancong - Historical Insight**  
Kue Pancong has a unique and interesting name. Literally, "pancong" comes from the word "cong," which is a Javanese term for "cup" or "small plate." This cake is known for its cup-like shape with a hole in the middle.

Although it's difficult to determine the exact origins of Kue Pancong, its history is believed to date back to the Dutch colonial era in Indonesia. It's said that this cake first appeared in the Batavia area (now Jakarta) in the 17th century, often sold by street vendors as a favorite snack for local people.
""",
        "ingredients": """
**Ingredients:**  
- 150g rice flour  
- 1 tbsp tapioca flour  
- 200g young grated coconut  
- 1 tsp salt  
- Granulated sugar to taste  
- 450ml water (or coconut milk)  
"""
    },
    "kue_rangi": {
        "history": """
**Kue Rangi - Historical Insight**  
Kue Rangi or Sagu Rangi is a traditional Betawi cake from Jakarta, made from a mixture of cornstarch (often called sago flour) and grated coconut, baked in special molds over a small stove and covered until evenly cooked.

Although specific historical information is limited, this cake has become part of Betawi's rich culinary heritage. Often described as Indonesia's signature coconut waffle, Kue Rangi has a crispy exterior and soft interior with a distinctive coconut flavor, served with palm sugar syrup on top.
""",
        "ingredients": """
**Ingredients:**  
- 100g grated coconut  
- 100g tapioca flour  
- 40ml coconut water  
- 1/4 tsp salt  

**Palm Sugar Sauce:**
- 75g palm sugar  
- 1/2 tbsp granulated sugar  
- 1/4 tsp salt  
- 125ml water  
- 1 pandan leaf, knotted  
- 1.5 tbsp tapioca flour + 2 tbsp water  
"""
    },
    "ongol_ongol": {
        "history": """
**Ongol-Ongol - Historical Insight**  
Ongol is one of the traditional cuisines of the Betawi people. "Ongol" is an archaic Betawi word meaning chewy. It can usually be found in traditional cultural events such as Betawi weddings or certain ceremonies.

According to history, based on stories from elders, the word "ongol" means chewy, flexible, and soft. Some assume that the creation of ongol originated from a type of cake often eaten by soldiers from Mongolia. Since people didn't know the name of this cake, they spontaneously called it "Ongol-ongol."
""",
        "ingredients": """
**Ingredients:**  
- 350g palm sugar  
- 1 1/2 tbsp granulated sugar  
- 1 pandan leaf  
- 400ml water  
- 375g tapioca flour  
- 1/2 tsp salt  
- 500ml coconut milk  
- 2 tbsp cooking oil  

**Coating:**
- 150g young grated coconut  
- 1/4 tsp salt  
- 1 pandan leaf  
"""
    },
    "putu_mayang": {
        "history": """
**Putu Mayang - Historical Insight**  
Putu Mayang is a traditional Betawi snack with deep cultural connections. Many historical references link its existence to Betawi folk tales, including the story of Jampang Mayangsari. The word "Mayang" is associated with the beautiful character "Mayang Sari."

In folklore, "Mayang" is described as something wavy, coiled, and beautiful, like the wavy and coiled shape of Putu Mayang resembling a scarf fluttering in the wind. Based on oral sources passed down through generations, Putu Mayang is closely related to Kue Mayam from South India, likely due to Batavia's position as an important port and trading center with multinational social interactions.
""",
        "ingredients": """
**Ingredients:**  
- 250g rice flour  
- 125g tapioca flour  
- 750ml coconut milk (for dough and kinca)  
- 3 pandan leaves  

**Kinca (Palm Sugar Sauce):**
- 75g palm sugar  
- 1 tbsp half-mature grated coconut  
- Pinch of salt  
- 2 tsp cornstarch + 3 tbsp water  
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

# Footer with app statistics
if uploaded_file:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="stats-card">
                <div class="feature-icon">🎯</div>
                <h4>Accuracy</h4>
                <p>High precision classification using ResNet50</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="stats-card">
                <div class="feature-icon">⚡</div>
                <h4>Fast Processing</h4>
                <p>Quick results in seconds</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="stats-card">
                <div class="feature-icon">🇮🇩</div>
                <h4>Indonesian Heritage</h4>
                <p>Preserving traditional snack knowledge</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

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
