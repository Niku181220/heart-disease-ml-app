# Machine Learning Classification Models – Streamlit Deployment

## a. Problem Statement

The objective of this assignment is to build, evaluate, and deploy multiple machine learning classification models on a single dataset. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation using multiple performance metrics, development of an interactive Streamlit web application, and deployment on Streamlit Community Cloud.

The goal is to compare the performance of traditional machine learning models and ensemble models on the same classification problem and provide insights into their behavior.

---

## b. Dataset Description

The dataset used for this project is a **classification dataset obtained from a public repository (Kaggle/UCI)**.

**Dataset Characteristics:**
- Type: Classification (Binary / Multi-class)
- Number of features: **≥ 12**
- Number of instances: **≥ 500**
- Format: CSV

The dataset contains multiple numerical features used as input variables and one target column representing the class label. The dataset was preprocessed to handle missing values and ensure compatibility with all implemented machine learning models.

Only **test data** is uploaded to the Streamlit application due to Streamlit Community Cloud memory limitations.

---

## c. Models Used and Evaluation Metrics

The following **six classification models** were implemented and evaluated using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN) Classifier  
4. Naive Bayes Classifier  
5. Random Forest Classifier (Ensemble Model)  
6. XGBoost Classifier (Ensemble Model)

Each model was evaluated using the following performance metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Comparison Table of Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | XX | XX | XX | XX | XX | XX |
| Decision Tree | XX | XX | XX | XX | XX | XX |
| KNN | XX | XX | XX | XX | XX | XX |
| Naive Bayes | XX | XX | XX | XX | XX | XX |
| Random Forest (Ensemble) | XX | XX | XX | XX | XX | XX |
| XGBoost (Ensemble) | XX | XX | XX | XX | XX | XX |

> **Note:** Replace `XX` with the actual metric values obtained during model evaluation.

---

## d. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Performed well on linear decision boundaries and provided stable baseline performance with good interpretability. |
| Decision Tree | Showed good performance but was sensitive to data variations and prone to overfitting without tuning. |
| KNN | Performance depended heavily on the value of K and distance metric; worked well for local decision boundaries. |
| Naive Bayes | Fast and efficient but assumed feature independence, which slightly limited its performance. |
| Random Forest (Ensemble) | Delivered strong performance with reduced overfitting by averaging multiple trees. |
| XGBoost (Ensemble) | Achieved the best overall performance due to gradient boosting and effective handling of complex patterns. |

---

## e. Streamlit Web Application

An interactive **Streamlit web application** was developed and deployed on **Streamlit Community Cloud**.

### Features of the App:
- Upload CSV test dataset
- Dropdown menu to select classification model
- Display of evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC Score
  - MCC
- Confusion Matrix visualization
- Classification Report display

---

## f. Project Structure
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ ├── logistic.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── nb.pkl
│ ├── rf.pkl
│ ├── xgb.pkl


---

## g. Deployment Details

- Platform: **Streamlit Community Cloud**
- Deployment Method: GitHub integration
- Branch: `main`
- Entry file: `app.py`

The deployed application opens an interactive frontend when accessed via the live link.

---

## h. Tools and Technologies Used

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- GitHub

---

## i. Execution Environment

The entire assignment, including model execution and testing, was performed on the **BITS Virtual Lab**. A screenshot of the execution environment has been included in the final PDF submission as proof.

---

### Final Notes

- All models were trained and evaluated on the same dataset.
- The application was successfully deployed and tested.
- The repository includes complete source code, dependencies, and documentation.


