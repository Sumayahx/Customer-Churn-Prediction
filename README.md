# Telecom Customer Churn Prediction App

An end-to-end Machine Learning web application that predicts customer churn for a telecommunications company. Built with Python, Scikit-Learn, and Streamlit, featuring robust data pipelines and hyperparameter tuning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-edscrvtxbwi5oezqdewecg.streamlit.app/)

---

## Project Overview
Customer churn is a critical challenge for subscription-based businesses. This project builds a production-ready machine learning pipeline to identify customers who are at high risk of canceling their service, allowing companies to target retention efforts proactively.

Unlike basic tutorial notebooks, this project implements **reusable Scikit-Learn pipelines** to prevent data leakage and is deployed as an interactive web app where users can input customer metrics and get real-time predictions.

---

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformers, GridSearchCV)
* **Models Evaluated:** Logistic Regression, Random Forest, Decision Tree, KNN, SVC
* **Deployment & UI:** Streamlit, Joblib, Git/GitHub

---

## Methodology & Modeling
1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions, numerical charges, and categorical variables against churn behavior.
2. **Preprocessing & Pipelines:** Used Scikit-Learn `ColumnTransformer` to handle missing values (`SimpleImputer`), scale numerical data (`StandardScaler`), and encode categorical text features (`OneHotEncoder`) seamlessly.
3. **Model Selection & Evaluation:** Trained and evaluated 5 different models. Because churn datasets are typically imbalanced, evaluation prioritized **ROC-AUC, Precision, and Recall** over raw accuracy.
   * *Winner:* **Logistic Regression** (Optimized via GridSearchCV to an ROC-AUC of **0.8461**).
4. **Serialization:** Saved the entire preprocessing and modeling pipeline into a single compressed object using `joblib` (`churn_pipeline.pkl`).

---

## How to Run Locally

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sumayahx/Customer-Churn-Prediction.git](https://github.com/Sumayahx/Customer-Churn-Prediction.git)
   cd Customer-Churn-Prediction
