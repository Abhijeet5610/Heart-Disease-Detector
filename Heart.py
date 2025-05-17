import os
import pickle
import numpy as np
from collections import Counter

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
    return Counter(predictions).most_common(1)[0][0]

def get_input(prompt, dtype=float, choices=None):
    while True:
        val = input(prompt)
        try:
            val = dtype(val)
            if choices and val not in choices:
                print(f"Invalid input. Expected one of {choices}. Try again.")
            else:
                return val
        except:
            print(f"Invalid input type. Expected {dtype.__name__}. Try again.")

def main():
    print("Enter patient details:")
    age = get_input("Age (years): ", int)
    sex = get_input("Sex (0=Female, 1=Male): ", int, choices=[0,1])
    cp = get_input("Chest Pain Type (0-3): ", int, choices=[0,1,2,3])
    trestbps = get_input("Resting Blood Pressure (mm Hg): ", float)
    chol = get_input("Cholesterol (mg/dl): ", float)
    fbs = get_input("Fasting Blood Sugar > 120 mg/dl (0 or 1): ", int, choices=[0,1])
    restecg = get_input("Resting ECG Results (0-2): ", int, choices=[0,1,2])
    thalach = get_input("Max Heart Rate Achieved: ", float)
    exang = get_input("Exercise Induced Angina (0 or 1): ", int, choices=[0,1])
    oldpeak = get_input("Oldpeak (ST depression): ", float)
    slope = get_input("Slope of Peak ST Segment (0-2): ", int, choices=[0,1,2])
    ca = get_input("Number of major vessels (0-4): ", int, choices=[0,1,2,3,4])
    thal = get_input("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect): ", int, choices=[1,2,3])

    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal])

    prediction = bagging_predict(trees, input_data)
    print("\nPrediction Result:")
    if prediction == 1:
        print(" This person is likely to have heart disease.")
    else:
        print(" This person is likely to be normal.")

if __name__ == "__main__":
    main()
