# 🏦 Bank Term Deposit Subscription Predictor

A machine learning web application that predicts whether a customer 
will subscribe to a bank term deposit based on demographic, contact, 
and economic data.

## 🚀 Live Demo
https://vinaygoud-bank-predictor.streamlit.app/

## 📊 Dataset
- Source: UCI ML Repository — Bank Marketing Dataset
- Records: 41,188 rows | 20 features

## 🛠️ Tech Stack
- Python, Scikit-learn, Random Forest Classifier
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit (deployed on Streamlit Cloud)

## ⚙️ ML Pipeline
- One Hot Encoding (9 categorical columns)
- MinMax Scaling (8 numerical columns)
- Random Forest — 100 estimators, random_state=42

## 📈 Features
- Real-time prediction with probability breakdown
- Confidence scoring (High / Medium / Low)
- Economic indicators integration
- Full input summary panel
