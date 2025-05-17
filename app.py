import streamlit as st
import numpy as np
import pickle
from collections import Counter
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best_random_forest.pkl')

with open(model_path, 'rb') as f:
    trees = pickle.load(f)

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    count = Counter(predictions)
    prediction, votes = count.most_common(1)[0]
    prob = votes / len(trees)
    return prediction, prob

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("🩺 Heart Disease Prediction Web App")
st.markdown("""
Enter the patient's medical details below.  
This model predicts whether the person is likely to have heart disease.
""")

age = st.number_input('Age (years)', min_value=1, max_value=120, value=54)
sex = st.selectbox('Sex', options=[0,1], format_func=lambda x: 'Female' if x==0 else 'Male')
cp = st.selectbox('Chest Pain Type (0-3)', options=[0,1,2,3], 
                  help='0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic')
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=140)
chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=239)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
restecg = st.selectbox('Resting ECG Results (0-2)', options=[0,1,2],
                       help='0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy')
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=160)
exang = st.selectbox('Exercise Induced Angina', options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=6.0, value=1.2, step=0.1)
slope = st.selectbox('Slope of the Peak ST Segment (0-2)', options=[0,1,2],
                     help='0=Upsloping, 1=Flat, 2=Downsloping')
ca = st.selectbox('Number of Major Vessels Colored (0-4)', options=[0,1,2,3,4])
thal = st.selectbox('Thalassemia', options=[1,2,3],
                    help='1=Normal, 2=Fixed Defect, 3=Reversible Defect')

input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal])

if st.button("Predict"):
    prediction, confidence = bagging_predict(trees, input_data)
    
    st.markdown("## Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ This person is likely to have **heart disease** with confidence {confidence:.2%}")
    else:
        st.success(f"✅ This person is likely to be **normal** with confidence {confidence:.2%}")
    
    labels = ['Heart Disease', 'Normal']
    values = [confidence, 1 - confidence] if prediction == 1 else [1 - confidence, confidence]
    
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['red', 'green'])
    ax.set_ylim([0,1])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
    st.pyplot(fig)

