import streamlit as st
import pickle
import numpy as np

# Load vectorizer and model
with open(r"D:\Buildables Internship\Cyber_Bullying\spam_api_bundle\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"D:\Buildables Internship\Cyber_Bullying\spam_api_bundle\model.pkl", "rb") as f:
    model = pickle.load(f)

# Define class labels (must match your training order)
class_labels = [
    "age_cyberbullying",
    "ethnicity_cyberbullying",
    "gender_cyberbullying",
    "not_cyberbullying",
    "other_cyberbullying",
    "religion_cyberbullying"
]

# Streamlit app UI
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🚨", layout="centered")

st.title("🚨 Cyberbullying Detection App")
st.write("Enter text below to check if it contains **cyberbullying** and what type it might be.")

# User input
user_text = st.text_area(" Enter your message here:", height=150)

if st.button("🔍 Analyze"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Transform text using the saved TF-IDF vectorizer
        X_input = vectorizer.transform([user_text])

        # Predict probabilities
        probs = model.predict_proba(X_input)[0]
        pred_class = np.argmax(probs)
        pred_label = class_labels[pred_class]
        confidence = probs[pred_class]

        # Display result
        if pred_label == "not_cyberbullying":
            st.success(f"✅ Prediction: **{pred_label}** (Confidence: {confidence:.2f})")
        else:
            st.error(f"🚨 Prediction: **{pred_label}** (Confidence: {confidence:.2f})")

        # Show all class probabilities
        st.subheader("📊 Class Probabilities")
        prob_dict = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
        st.json(prob_dict)