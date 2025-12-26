# AutoJudge – Programming Problem Difficulty Predictor

AutoJudge is a machine learning–based system that predicts the **difficulty of programming problems** using only their textual descriptions.  
It outputs both a **difficulty class (Easy / Medium / Hard)** and a **continuous difficulty score**, emulating how online judges assess problem complexity.

This project was developed as part of the **ACM IIT Roorkee Open Projects** initiative.

---

##  Features

- Text-only difficulty prediction
- Difficulty **classification** (Easy / Medium / Hard)
- Difficulty **regression score** (continuous)
- TF-IDF based semantic feature extraction
- Handcrafted complexity signals
- Confidence-aware interpretation of predictions
- Interactive Streamlit web interface
- No authentication or database required

---

## Problem Overview

Online judges and learning platforms need to estimate the difficulty of programming problems before publishing them.  
AutoJudge approaches this problem using **only the problem statement**, without executing code or analyzing solutions.

The system predicts:
- A **categorical difficulty label**
- A **numerical difficulty score** indicating relative complexity

This mirrors real-world ambiguity, where problem difficulty is not always sharply defined.

---

## Methodology

###  Text Processing (Fully Implemented)

The following fields are **combined into a single text input**, exactly as used in the code:

- Problem description
- Input description
- Output description

Minimal preprocessing is applied to preserve semantic structure.

---

### Feature Engineering

#### 1.TF-IDF Features
TF-IDF vectorization is used to capture semantic and contextual information from problem text.

- Separate TF-IDF vectorizers are trained for:
  - Classification
  - Regression
- This avoids feature-space mismatch during inference.

#### 2.Handcrafted Features
To complement TF-IDF, additional complexity signals are extracted:

- **Text length** (proxy for problem detail)
- **Mathematical symbol count** (`+ - * / < >`)
- **Algorithmic keyword frequency**  
  (`graph`, `dp`, `recursion`, `tree`, `greedy`)

These features capture structural and algorithmic cues often associated with higher difficulty.

---

## 🤖 Models and Experiments

### Difficulty Classification (Easy / Medium / Hard)

The following models were evaluated:

| Model | Observed Accuracy | Notes |
|----|----|----|
| Logistic Regression | ~44% | Baseline, underfits complex patterns |
| Linear SVM | ~45–47% | Sensitive to feature scaling |
| Random Forest Classifier | **~50% (best)** | Handles non-linear feature interactions well |

#### Final Choice: **Random Forest Classifier**
- Robust to feature heterogeneity
- Better handles class imbalance
- Improved performance with handcrafted features
- More stable predictions across classes

---

### Difficulty Score Regression

The following regression models were evaluated:

| Model | RMSE (approx.) | Notes |
|----|----|----|
| Linear Regression | High error | Unable to model non-linear difficulty trends |
| Random Forest Regressor | ~2.04 | Strong but slightly overfits |
| Gradient Boosting Regressor | **~2.01 (best)** | Best bias–variance tradeoff |

####  Final Choice: **Gradient Boosting Regressor**
- Captures non-linear difficulty progression
- Lower RMSE than alternatives
- More stable generalization

---

## 📊 Output Interpretation

### Difficulty Class
One of:
- `Easy`
- `Medium`
- `Hard`

### Difficulty Score
A continuous numerical value representing relative difficulty.

> Borderline cases (e.g., Easy with a higher score) are expected and reflect real-world ambiguity in problem difficulty rather than model error.

Classification and regression models are intentionally **kept independent** to preserve this nuance.

---

## Web Interface (Streamlit)

The Streamlit UI allows users to:
1. Paste problem description
2. Paste input description
3. Paste output description
4. Click **Predict Difficulty**
5. View:
   - Predicted difficulty class
   - Predicted difficulty score

The UI is designed for simplicity and evaluator usability.

---


