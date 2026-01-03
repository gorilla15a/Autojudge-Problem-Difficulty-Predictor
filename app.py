import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# LOAD MODELS 
rf_clf = joblib.load("models/rf.pkl")
gb_reg = joblib.load("models/gb_regressor.pkl")

tfidf_clf = joblib.load("models/tfidf_classifier.pkl")
tfidf_reg = joblib.load("models/tfidf_regressor.pkl")

scaler = joblib.load("models/scaler.pkl")

# FEATURE FUNCTIONS 
math_symbols = ['+', '-', '*', '/', '%', '<', '>', '=', '^']
keywords = [
    'graph', 'dp', 'dynamic programming', 'recursion',
    'optimize', 'complexity', 'tree', 'greedy'
]

def count_math_symbols(text):
    return sum(text.count(sym) for sym in math_symbols)

def keyword_frequency(text):
    return sum(text.count(k) for k in keywords)

def semantic_engineered_features(text):
    return np.array([
        len(text),
        count_math_symbols(text),
        keyword_frequency(text)
    ])

def extract_structural_features(sample_io):
    text = str(sample_io)

    numbers = re.findall(r'-?\d+', text)
    num_numbers = len(numbers)
    max_num_length = max([len(n) for n in numbers], default=0)

    num_lines = text.count('\n') + 1 if text.strip() else 0
    num_brackets = sum(text.count(b) for b in ['[', ']', '{', '}', '(', ')'])

    has_matrix_like = int(
        '[' in text and ']' in text and ',' in text and '\n' in text
    )

    return np.array([
        num_numbers,
        num_lines,
        num_brackets,
        max_num_length,
        has_matrix_like
    ])

# STREAMLIT UI 
st.title("AutoJudge – Problem Difficulty Predictor")

problem_desc = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")
sample_io = st.text_area("Sample Input / Output (optional)")

if st.button("Predict Difficulty"):

    # Combine semantic text 
    full_text = problem_desc + " " + input_desc + " " + output_desc

    # Engineered features (shared) 
    sem_feats = semantic_engineered_features(full_text)
    struct_feats = extract_structural_features(sample_io)

    engineered = np.concatenate([sem_feats, struct_feats]).reshape(1, -1)
    engineered_scaled = scaler.transform(engineered)

    # CLASSIFICATION PIPELINE 
    X_tfidf_clf = tfidf_clf.transform([full_text])
    X_final_clf = hstack([X_tfidf_clf, engineered_scaled]).toarray()

    difficulty_class = rf_clf.predict(X_final_clf)[0]

    # REGRESSION PIPELINE
    X_tfidf_reg = tfidf_reg.transform([full_text])
    X_final_reg = hstack([X_tfidf_reg, engineered_scaled]).toarray()

    difficulty_score = gb_reg.predict(X_final_reg)[0]

    # Display 
    st.subheader("Prediction Result")
    st.write(f"**Predicted Difficulty Class:** {difficulty_class.capitalize()}")
    st.write(f"**Predicted Difficulty Score:** {difficulty_score:.2f}")
