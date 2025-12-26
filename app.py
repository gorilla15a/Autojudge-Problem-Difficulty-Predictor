import streamlit as st
import numpy as np
import joblib
import os

# --------------------------------------------------
# Safety check: required files
# --------------------------------------------------
REQUIRED_FILES = [
    "rf.pkl",
    "tfidf_classifier.pkl",
    "gb_regressor.pkl",
    "tfidf_regressor.pkl",
    "scaler.pkl"
]

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        st.error(f"Missing required file: {file}")
        st.stop()

# --------------------------------------------------
# Load trained models and preprocessors
# --------------------------------------------------
rf_clf = joblib.load("rf.pkl")
tfidf_clf = joblib.load("tfidf_classifier.pkl")

gb_reg = joblib.load("gb_regressor.pkl")
tfidf_reg = joblib.load("tfidf_regressor.pkl")

scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------
st.set_page_config(
    page_title="AutoJudge – Problem Difficulty Predictor",
    layout="centered"
)

st.title(" AutoJudge")
st.write("Predict the difficulty of a programming problem")

# --------------------------------------------------
# User inputs
# --------------------------------------------------
problem_desc = st.text_area(
    "Problem Description",
    height=200,
    placeholder="Paste the full problem statement here..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Describe the output format..."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Difficulty"):

    if problem_desc.strip() == "":
        st.warning("Please enter the problem description.")
        st.stop()

    # Combine text
    full_text = problem_desc + " " + input_desc + " " + output_desc

    # -------------------------
    # Handcrafted features
    # -------------------------
    length = len(full_text)
    symbol_count = sum(full_text.count(s) for s in ['+', '-', '*', '/', '%', '<', '>'])
    keyword_count = sum(
        full_text.count(k)
        for k in ['graph', 'dp', 'recursion', 'tree', 'greedy']
    )

    handcrafted = np.array([[length, symbol_count, keyword_count]])
    handcrafted = scaler.transform(handcrafted)

    # -------------------------
    # Classification
    # -------------------------
    vec_clf = tfidf_clf.transform([full_text])
    features_clf = np.hstack([vec_clf.toarray(), handcrafted])
    difficulty_class = rf_clf.predict(features_clf)[0]

    # -------------------------
    # Regression
    # -------------------------
    vec_reg = tfidf_reg.transform([full_text])
    features_reg = np.hstack([vec_reg.toarray(), handcrafted])
    difficulty_score = gb_reg.predict(features_reg)[0]

    # -------------------------
    # Output
    # -------------------------
    st.success("Prediction complete ")
    st.markdown(f"###  Difficulty Class: **{difficulty_class.upper()}**")
    st.markdown(f"###  Difficulty Score: **{round(float(difficulty_score), 2)}**")
