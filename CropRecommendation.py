import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained XGBoost model
with open("xgboost_crop_recommendation.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the crop information dataset
crop_info = pd.read_csv("crop_info.csv")

# Create a list of crops if the model returns a numerical index
crop_names = crop_info["Crop"].tolist()

# Streamlit UI
st.title("Crop Recommendation System 🌱")
st.write("Enter the soil and weather conditions to get the best crop recommendation.")

# Input fields with unique keys
N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50, key="N_input")
P = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50, key="P_input")
K = st.number_input("Potassium (K)", min_value=0, max_value=150, value=50, key="K_input")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, key="temp_input")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, key="humidity_input")
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, key="ph_input")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, key="rainfall_input")

# Predict the crop
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]

    # If the model returns a numerical index, map it to the crop name
    if isinstance(prediction, (int, np.integer)):
        prediction = crop_names[prediction]

    # Ensure prediction is treated as a string before performing string operations
    prediction = str(prediction).strip().lower().capitalize()

    # Get crop details from crop_info
    crop_details = crop_info[crop_info["Crop"].str.strip().str.lower() == prediction.lower()]
    
    st.success(f"Recommended Crop: **{prediction}** 🌾")

    if not crop_details.empty:
        st.subheader("Crop Information:")
        st.write(f"**Methods:** {crop_details['Methods'].values[0]}")
        st.write(f"**Steps:** {crop_details['Steps'].values[0]}")
        st.write(f"**Environment:** {crop_details['Environment'].values[0]}")
    else:
        st.warning("No additional information available for this crop.")

