import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. PAGE CONFIGURATION & ANIMATED BACKGROUND ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

def set_custom_style():
    """Injects custom CSS for background image and animations."""
    # Using a high-quality Unsplash wine image for the background
    background_url = "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=2070&auto=format&fit=crop"
    
    custom_css = f"""
    <style>
    /* Background Image setup */
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{background_url}");
        background-attachment: fixed;
        background-size: cover;
    }}
    
    /* CSS Animation for the title */
    @keyframes floatDown {{
        0% {{ opacity: 0; transform: translateY(-50px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Apply styling to headings and text to make them readable against dark background */
    h1 {{
        animation: floatDown 1.2s ease-out;
        color: #f8c291 !important; /* Soft wine/gold color */
        text-align: center;
        text-shadow: 2px 2px 5px #000;
    }}
    h3, p, label {{
        color: white !important;
        text-shadow: 1px 1px 3px #000;
    }}
    
    /* Style the prediction box */
    .prediction-box {{
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
        text-align: center;
        border: 1px solid #f8c291;
        animation: floatDown 0.8s ease-out;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_style()

# --- 2. LOAD ML COMPONENTS ---
@st.cache_resource
def load_models():
    """Loads the saved models from your Jupyter Notebook session."""
    try:
        model = joblib.load("best_model.pkl") # Your saved SVC
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("feature_columns.pkl")
        return model, scaler, features
    except FileNotFoundError:
        st.error("⚠️ Error: Missing .pkl files. Ensure 'best_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the same folder as app.py.")
        st.stop()

model, scaler, feature_names = load_models()

# --- 3. FRONT FACE UI & INPUTS ---
st.markdown("<h1>🍷 AI Wine Quality Sommelier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the chemical properties below to predict the quality of the wine.</p>", unsafe_allow_html=True)
st.markdown("---")

# Dynamically create input fields based on your saved features
st.markdown("### 🧪 Wine Characteristics")
input_data = {}

# Create 2 columns for a cleaner layout
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    # Alternate between columns for layout
    with col1 if i % 2 == 0 else col2:
        # Defaulting step to 0.01 for smooth slider/number input
        input_data[feature] = st.number_input(f"{feature}", value=0.00, step=0.01)

# --- 4. PREDICTION LOGIC & ANIMATIONS ---
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🔮 Predict Wine Quality", use_container_width=True)

if predict_button:
    # Convert user input into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input using your saved scaler
    scaled_input = scaler.transform(input_df)
    
    # Make prediction using your saved SVM
    prediction = model.predict(scaled_input)[0]
    
    # Success Animation & Output
    st.balloons() # Built-in Streamlit animation!
    
    st.markdown(
        f"""
        <div class="prediction-box">
            <h2 style='color: white;'>Predicted Quality Score</h2>
            <h1 style='color: #44bd32; font-size: 50px;'>{prediction}</h1>
            <p>Based on the Support Vector Classifier</p>
        </div>
        """, 
        unsafe_allow_html=True
    )