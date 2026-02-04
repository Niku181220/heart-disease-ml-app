import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Prediction App")
st.write("Upload a CSV of test samples and choose a model to predict heart disease risk.")

# Model dropdown
model_choice = st.selectbox(
    "Select a Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

model_map = {
    "Logistic Regression": "logistic_reg.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    # Show raw uploaded data
    st.write("Uploaded Data (Raw):")
    st.write(test_df.head())

    # Remove target column if present
    if "target" in test_df.columns:
        st.warning("Target column detected — removing it before prediction.")
        test_df = test_df.drop("target", axis=1)

    # Columns used during model training
    required_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal'
    ]

    # Ensure correct column order
    test_df = test_df[required_cols]

    st.write("Uploaded Data (Processed):")
    st.write(test_df.head())

    # Load selected model
    model = pickle.load(open(f"models/{model_map[model_choice]}", "rb"))

    # Make Predictions
    predictions = model.predict(test_df)

    st.subheader("Predictions")
    st.write(predictions)

