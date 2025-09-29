import streamlit as st
import pickle
import numpy as np
import os

# Load model files
with open(os.path.join(os.path.dirname(__file__), "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb") as f:
    model = pickle.load(f)

class_labels = ["age_cyberbullying", "ethnicity_cyberbullying", "gender_cyberbullying", "not_cyberbullying", "other_cyberbullying", "religion_cyberbullying"]

st.set_page_config(page_title="Cyberbullying Detector", page_icon="🚨", layout="wide")

# Read and include external CSS file
def load_css(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Load CSS from the static folder
css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
try:
    css_content = load_css(css_path)
    st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Please ensure styles.css exists in the static folder.")

st.markdown('<h1 class="header">🚨 CYBER-BULLYING DETECTION</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 📝 Enter text to analyze:")
    user_text = st.text_area("", height=100, label_visibility="collapsed", placeholder="Type your text here...")

    if st.button("🔍 Analyze Text", help="Click to analyze the text"):
        if user_text.strip():
            X_input = vectorizer.transform([user_text])
            probs = model.predict_proba(X_input)[0]
            pred_class = np.argmax(probs)
            pred_label = class_labels[pred_class]
            confidence = probs[pred_class]

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="analysis-results">', unsafe_allow_html=True)

            st.write("### 📄 Original Text:")
            st.info(f'"{user_text}"')

            st.write("### 🔍 Analysis Result:")
            # Confidence score hidden from user
            if pred_label == "not_cyberbullying":
                st.markdown('<div class="success">✅ **SAFE CONTENT**</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error">🚨 **{pred_label.replace("_", " ").title()}**</div>', unsafe_allow_html=True)

            # Confidence score and progress bar hidden from user
            # st.write("**Confidence Score:**")
            # st.progress(confidence)

            st.write("**📊 All Class Probabilities:**")
            for label, prob in zip(class_labels, probs):
                st.write(f"**{label.replace('_', ' ').title()}:** {prob*100:.1f}%")
                st.progress(prob)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="analysis-results">', unsafe_allow_html=True)
            st.warning("⚠️ Please enter some text to analyze.")
            st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="how-to-use">', unsafe_allow_html=True)
    st.write("### 🟢 How to Use:")
    st.write("1️⃣ **Enter text** above")
    st.write("2️⃣ **Click Analyze**")
    st.write("3️⃣ **Get results** instantly")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="model-info">', unsafe_allow_html=True)
    st.write("### 💡 Model Info:")
    st.write("• **6 detection classes**")
    st.write("• **High accuracy**")
    st.write("• **Real-time analysis**")
    st.markdown('</div>', unsafe_allow_html=True)
