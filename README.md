# AutoJudge – Programming Problem Difficulty Predictor

AutoJudge is a machine learning–based system that predicts the **difficulty of programming problems** using only their textual descriptions.  
It outputs both a **difficulty class (Easy / Medium / Hard)** and a **continuous difficulty score**, emulating how online judges assess problem complexity.


---

##  Features

- Text-only difficulty prediction
- Difficulty **classification** (Easy / Medium / Hard)
- Difficulty **regression score** Range between 1 to 10.
- TF-IDF based semantic feature extraction
- Engineered features based on semantic description like length of full text,frequency of mathematical operators and frequency of algorithmic keywords like    DP,graph and structual features of sample input and output.
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
## Dataset Used
The Dataset used is the Task complexity Dataset the one mentioned in the problem statement word document. The dataset consists of 4112 rows and 8 columns.
The columns are as follows:
Index(['title', 'description', 'input_description', 'output_description',
       'sample_io', 'problem_class', 'problem_score', 'url'],
      dtype='object')

---

## Methodology
The methodology involves combining the input decription,output description and description of problem as full text and then using tf-idf importanat features are extracted and also semantic & structural features are also extracted to compliment tf-idf all these are used for model training. Later in web interface prediction also the inputs from the user are combined and then the tfidf implemenatation , semantic and structural features are all combined as input for the final prediciton for problem class and problem score.

###  Data Preprocessing & Feature Extraction
This involves cleaning the text and replacing nan string with spaces,list and dictionaries are saved in string form and all strings are saved in lower form.The problem description,input description and output description are all combined to form the column full text which will serve as the test datatset.

#### 1.TF-IDF Features
TF-IDF vectorization is used to capture semantic and contextual information from problem text.TF-IDF encodes contextual importance by assigning higher weights to words that are frequent in a specific problem but rare across the dataset, allowing models to learn topic- and task-relevant patterns rather than raw word counts.

- Separate TF-IDF vectorizers are trained for:
  - Classification
  - Regression
- This avoids feature-space mismatch during inference.

#### 2.Engineered Features
To complement TF-IDF, additional complexity signals are extracted:

- **Text length** (proxy for problem detail)
- **Mathematical symbol count** (`+ - * / < >`)
- **Algorithmic keyword frequency**  
  (`graph`, `dp`, `recursion`, `tree`, `greedy`)

These features capture structural and algorithmic cues often associated with higher difficulty.

- **Text Length** (Time complexity proxy of Sample Input & Output)
- **has_Matrix like**
---

## 🤖 Models and Experiments

### Difficulty Classification (Easy / Medium / Hard)

The following models were evaluated:

| Model | Observed Accuracy | Notes |
|----|----|----|
| Logistic Regression | 44.8% | Baseline, underfits complex patterns |
| Linear SVM | 48.11 | Sensitive to feature scaling |
| Random Forest Classifier | **53.1%** | Handles non-linear feature interactions well |

#### Final Choice: **Random Forest Classifier**
- Robust to feature heterogeneity
- Better handles class imbalance
- Improved performance with handcrafted features
- More stable predictions across classes
The confusion matrix are attached in the report.
---

### Difficulty Score Regression

The following regression models were evaluated:

| Model | RMSE | MAE | Notes |
|----|----|----|----|
| Linear Regression | 6.840915264460398 | 5.247318164914233 | Unable to model non-linear difficulty trends |
| Random Forest Regressor | 2.0506899051467267 | 1.710042519125868 | Strong but slightly overfits |
| Gradient Boosting Regressor | 1.6933092417895048 | 2.0348564071782165 | Best bias–variance tradeoff |

####  Final Choice: **Gradient Boosting Regressor**
- Captures non-linear difficulty progression
- Lower RMSE and MAE than alternatives
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

Classification and regression models are intentionally **kept independent** to preserve this nuance.More about this is explained in report.

---

## Web Interface (Streamlit)

The Streamlit UI allows users to:
1. Paste problem description
2. Paste input description
3. Paste output description
4. Paste sample input and output (Optional)
5. Click **Predict Difficulty**
6. View:
   - Predicted difficulty class
   - Predicted difficulty score

The UI is designed for simplicity and evaluator usability.

---

## Steps to Run locally

1.Create a new folder and select it n VScode to clone 

Clone the repo using the code in VS Code editor.

git clone https://github.com/gorilla15a/AutoJudge-Problem-Difficulty-Predictor.git

cd AutoJudge-Problem-Difficulty-Predictor

2.Create Virtual environment and activate it. Example as follows:

python -m venv vdemo

venv\Scripts\activate

3.Install the requirements as follows.

pip install -r requirements.txt

4.streamlit run app.py

---
## Video Demo Link

🔗 [Video-Demo](https://drive.google.com/drive/folders/17D8s8VWoIe--yQt4xFSGNPeY24fwgJEc?usp=drive_link)

 Access video using IITR email.

## Personal Info

Name: Ansul

E-mail:ansul@me.iitr.ac.in

Enrollment No:23113029


