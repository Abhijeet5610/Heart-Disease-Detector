# Heart Disease Prediction Using Scratch-built Machine Learning Models

## Project Overview
This project focuses on building machine learning models from scratch to predict the presence of heart disease based on patient medical data. The dataset contains various clinical attributes, and the goal is to classify whether a patient has heart disease (binary classification).

## Models Implemented
- Logistic Regression (from scratch)
- K-Nearest Neighbors (KNN) (from scratch)
- Decision Tree (from scratch)
- Random Forest (from scratch) **Final chosen model**

## Dataset
The data consists of clinical parameters such as age,sex,chest pain type,resting bp s,cholesterol,fasting blood sugar,resting ecg,max heart rate,exercise angina and more. Categorical variables were one-hot encoded for model compatibility.

## Data Preprocessing
- Handled categorical variables using one-hot encoding (`pd.get_dummies` with `drop_first=True` to avoid multicollinearity)
- Normalized numerical features
- Split dataset into training (80%) and testing (20%) sets

## Model Selection and Evaluation
Each model was trained and evaluated using metrics:
- Accuracy
- Precision
- Recall
- F1 Score

After extensive hyperparameter tuning and evaluation, the **Random Forest** model demonstrated the best overall performance with an F1 score of **0.9438** on the test data.

### Hyperparameters of Final Random Forest Model
- Number of Trees: 15
- Max Depth: 15
- Minimum Samples per Leaf: 1
- Sample Size (Bootstrap fraction): 1.0

## Usage Instructions

### 1. Training
- Training code is located in `model/train_model.py`
- To retrain, run the training script with desired parameters

### 2. Saving the Model
- The trained Random Forest model (list of trees) is saved using Python’s `pickle` as `model/best_random_forest.pkl`

### 3. Prediction
- Use `app/predict.py` to load the saved model and predict heart disease for new patient data
- Modify the input feature vector in `predict.py` to test different cases
- Example output:

### 4.Deployment
- The prediction script was then integrated into a **streamlit** or **Flask web app** for user-friendly interaction.

### 5.Folder Structure:
    Heart Disease Detector/
    ├── data/ # Dataset files
    ├── model/ # Training scripts and saved models
    ├── app/ # Prediction scripts and web app code
    ├── notebooks/ # Exploration and experiments
    ├── report/ # Documentation and reports
    ├── README.md # This file
    ├── requirements.txt # Python dependencies

### 6.Future Work
    This project lays a strong foundation for heart disease prediction using scratch-built machine learning models. Here are some meaningful next steps to enhance and extend the project:

    a. Feature Importance and Explainability:  
        Implement techniques like SHAP values or LIME to highlight which clinical features most influence predictions. This will increase trust and interpretability for medical professionals.

    b. Interactive Web Application:  
        Develop a user-friendly web app using Streamlit or Flask that allows users to input patient details and receive real-time predictions with clear explanations and visualizations.

    c. Cross-Validation and Robust Evaluation:  
        Incorporate k-fold cross-validation to ensure more reliable and stable model performance estimates, reducing risks of overfitting.

    d. Advanced Ensemble Models: 
        Explore and benchmark state-of-the-art ensemble algorithms such as XGBoost, LightGBM, or CatBoost to potentially improve accuracy and training speed.

    e. Multi-Class Classification or Severity Prediction:  
        Extend the model to predict different types or severity levels of heart disease, offering more granular diagnostic support.

    f. Integration with Electronic Health Records (EHR):  
        Investigate possibilities to connect the model with real-world medical record systems to facilitate clinical deployment.

    g. Cloud Deployment:  
        Host the prediction model and app on cloud platforms like Heroku, AWS, or Google Cloud to enable scalable, on-demand access.

    h. Model Monitoring and Updating: 
        Set up pipelines to monitor model predictions in production and periodically retrain the model with new incoming data to maintain accuracy over time.



    These future directions will help make the model more reliable, interpretable, and clinically relevant — paving the way towards real-world impact.

### 7.Acknowledgments

- Thanks to the Unified Mentor for providing the Dataset.
- Inspired by various machine learning tutorials and resources like Deep Learning.AI,etc.


## Contact

- For questions or collaboration, please reach out at:  iamabhijeet5610.2@gmail.com 

