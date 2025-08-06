import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load("cancer_model.pkl")
data = load_breast_cancer()
features = data.feature_names

# Title
st.title("🧬 Cancer Prediction AI Agent")
st.markdown("### Enter the required medical values to check cancer risk")

# Input from user
user_input = []
for feature in features:
    value = st.slider(feature, float(data.data[:, features.tolist().index(feature)].min()), 
                      float(data.data[:, features.tolist().index(feature)].max()))
    user_input.append(value)

# Prediction
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    if prediction == 0:
        st.error("⚠️ Prediction: Malignant (High Risk)")
        st.write("🩺 Recommendation: Please consult a doctor immediately. Avoid smoking, maintain a healthy weight, and schedule screening.")
    else:
        st.success("✅ Prediction: Benign (Low Risk)")
        st.write("👍 Recommendation: Continue healthy lifestyle, regular checkups, and stay physically active.")
