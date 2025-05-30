import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# Set page configuration
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# Title and description
st.title("🌱 Crop Recommendation System")
st.write("Enter the soil and environmental parameters to get the best crop recommendation.")

# Load pre-trained model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    model = load_model('crop_predictor.h5')
    scaler = joblib.load('scaler.pickle')
    label_encoder = joblib.load('label_encoder_crop.pickle')
    return model, scaler, label_encoder

# Load the model, scaler, and label encoder
model, scaler, label_encoder = load_model_and_preprocessors()

# Create input form
st.subheader("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=80.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)

# Prediction button
if st.button("Predict Crop"):
    # Prepare input data
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_crop = label_encoder.inverse_transform(predicted_class)[0]
    
    # Display result
    st.success(f"Recommended Crop: **{predicted_crop.capitalize()}**")
    
    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    probabilities = pd.DataFrame(
        prediction[0],
        index=label_encoder.classes_,
        columns=['Probability']
    )
    probabilities = probabilities.sort_values(by='Probability', ascending=False)
    st.bar_chart(probabilities)

# Add some styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)