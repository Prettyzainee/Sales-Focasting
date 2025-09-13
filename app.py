# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# === FILE PATHS ===
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.save"

# === LOAD MODEL & SCALER ===
@st.cache_resource
def load_model_and_scaler():
    #model = tf.keras.models.load_model(MODEL_PATH)
    model = tf.keras.models.load_model("model.h5", compile=False)

    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# Try to get feature names from scaler (if fitted with DataFrame)
if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_.tolist()
else:
    # fallback: manual input
    feature_names = ["current/discounted_price", "rating", "number_of_reviews"]

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="Sales Forecasting App", layout="wide")
st.title("📊 Sales, Demand & Discount Impact Forecasting")
st.markdown(
    """
    This interactive tool uses a **Neural Network model** trained on product data  
    to forecast **listed price / demand**, analyze **discount impact**, and visualize results.  
    """
)

# === SIDEBAR INPUT ===
st.sidebar.header("🔧 Enter Product Features")
user_input = {}

for feature in feature_names:
    # Choose default values depending on name
    default_val = 100.0
    if "rating" in feature.lower():
        default_val = 4.0
    elif "review" in feature.lower():
        default_val = 50.0
    elif "price" in feature.lower():
        default_val = 100.0

    user_input[feature] = st.sidebar.number_input(
        f"{feature}", min_value=0.0, max_value=1e6, value=float(default_val), step=1.0
    )

# Convert to dataframe
input_df = pd.DataFrame([user_input])

# === PREDICTION ===
if st.sidebar.button("🔮 Predict"):
    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input).flatten()[0]

    st.subheader("📈 Prediction Result")
    st.success(f"**Predicted Listed Price / Sales Value:** {prediction:.2f}")

    # === DISCOUNT IMPACT (if discount column exists) ===
    discount_cols = [f for f in feature_names if "discount" in f.lower()]
    if discount_cols:
        dcol = discount_cols[0]
        discount_value = input_df[dcol].iloc[0]
        impact = prediction - discount_value
        st.info(
            f"💡 Discount Analysis: With a discount price of {discount_value:.2f}, "
            f"the model predicts an impact of **{impact:.2f}** compared to listed price."
        )

    # === VISUALIZATION ===
    st.subheader("📊 Forecast Visualization")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Predicted"], [prediction], color="skyblue", label="Forecast")
    if discount_cols:
        ax.bar(["Discounted"], [discount_value], color="orange", label="Discount")
    ax.set_ylabel("Price / Sales Value")
    ax.legend()
    st.pyplot(fig)

# === SIDEBAR INFO ===
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ⚡ **App Features**
    - Predict sales / listed price
    - Auto-detects input features
    - Analyze discount impact
    - Visualize results with charts
    - Easy and interactive UI
    """
)
