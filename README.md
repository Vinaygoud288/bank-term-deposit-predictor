# 🏦 Bank Term Deposit Subscription Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-purple)

## 🚀 Live Demo

👉 **Try the Application:**  
**https://vinaygoud-bank-predictor.streamlit.app/**

---

## 📌 Project Overview

This project predicts whether a customer is likely to subscribe to a **bank term deposit** using machine learning. It follows an end-to-end workflow, including data preprocessing, exploratory data analysis, feature engineering, model building, hyperparameter tuning, and deployment using Streamlit.

The objective is to help banks identify potential customers who are more likely to subscribe, enabling targeted marketing campaigns and reducing unnecessary marketing costs.

---

## 🎯 Problem Statement

Direct marketing campaigns are expensive and time-consuming. Contacting every customer does not guarantee success.

The goal of this project is to build a predictive model that helps banks identify customers with a high probability of subscribing to a term deposit.

---

## 📊 Dataset Information

- **Dataset:** Bank Marketing Dataset
- **Source:** UCI Machine Learning Repository
- **Records:** 41,188
- **Features:** 20
- **Target Variable:** `y`

Target Classes:

- **Yes** → Customer subscribed
- **No** → Customer did not subscribe

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-Learn (SMOTE)
- Optuna
- Joblib
- Streamlit

---

## 📂 Project Structure

```
Bank-Term-Deposit-Prediction/
│
├── Bank_Term_Deposit_Prediction.ipynb
├── app.py
├── model_prod_files.pkl
├── requirements.txt
├── README.md
```

---

## 🔄 Machine Learning Workflow

### 1. Data Understanding

- Loaded dataset
- Examined data structure
- Checked feature types
- Statistical summary

---

### 2. Data Cleaning

- Checked duplicate records
- Verified missing values
- Removed unnecessary features (if applicable)

---

### 3. Exploratory Data Analysis (EDA)

Performed detailed analysis to understand:

- Customer demographics
- Marketing campaign patterns
- Economic indicators
- Subscription distribution
- Feature relationships

---

### 4. Feature Engineering

Created a new feature:

**Previously Contacted**

```
If previous > 0
    Previously Contacted = 1
Else
    Previously Contacted = 0
```

---

### 5. Data Preprocessing

Categorical Features

- One-Hot Encoding

Numerical Features

- Min-Max Scaling

Target Variable

- Label Encoding

---

### 6. Handling Class Imbalance

Since the dataset is imbalanced, SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the training data before model training.

---

### 7. Model Building

Multiple machine learning models were trained and compared.

The final selected model:

✅ Random Forest Classifier

---

### 8. Hyperparameter Optimization

Optuna was used to optimize the Random Forest model.

Optimized hyperparameters include:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

---

### 9. Model Evaluation

The final model was evaluated using:

- Accuracy

---

### 10. Model Deployment

The trained model and preprocessing objects were saved using Joblib.

Saved artifacts include:

- Random Forest Model
- OneHotEncoder
- MinMaxScaler
- LabelEncoder

The model was deployed as an interactive Streamlit web application.

---

## 📈 Prediction Pipeline

```
Customer Input
       │
       ▼
Feature Engineering
       │
       ▼
One-Hot Encoding
       │
       ▼
Min-Max Scaling
       │
       ▼
Feature Alignment
       │
       ▼
Random Forest Classifier
       │
       ▼
Prediction
       │
       ▼
Subscription Probability
```

---

## 🚀 Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/yourusername/Bank-Term-Deposit-Prediction.git
```

### Navigate to the project directory

```bash
cd Bank-Term-Deposit-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 💼 Business Impact

This solution helps banks:

- Improve marketing campaign efficiency
- Reduce customer acquisition costs
- Identify high-potential customers
- Increase campaign conversion rates
- Support data-driven decision making

---

## 🔮 Future Improvements

- XGBoost and LightGBM implementation
- SHAP explainability
- Probability calibration
- Docker containerization
- REST API using FastAPI
- Cloud deployment (AWS/Azure/GCP)
- Automated model monitoring

---

## 📚 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Feature Scaling
- Handling Imbalanced Data (SMOTE)
- Machine Learning
- Hyperparameter Optimization (Optuna)
- Model Evaluation
- Model Serialization
- Streamlit Deployment
- End-to-End Machine Learning Pipeline

---

## 📦 Requirements

Main libraries used:

- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn
- optuna
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Vinay Goud**

Machine Learning | Data Analytics | Python | SQL | Power BI

- GitHub: https://github.com/Vinaygoud288
- LinkedIn: https://www.linkedin.com/in/muthyala-vinay

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Feedback and contributions are always welcome!
