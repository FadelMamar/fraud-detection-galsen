# 🛡️ Financial Fraud Detection for Xente

This repository contains a machine learning solution for detecting **fraudulent financial transactions** in the context of the **Xente** e-commerce and financial services platform, which serves over 10,000 customers in Uganda.

## 📊 Competition Overview

**Xente** provided a dataset of \~140,000 anonymized transactions spanning from **15 November 2018 to 15 March 2019**. The goal of this project is to build a predictive model that accurately identifies **fraudulent activity**, thereby enhancing financial security and trust for Xente’s customers.

Fraud detection is a high-impact application of machine learning, particularly in the fintech sector. An effective fraud detection model can minimize financial loss, reduce risk, and provide a safer digital experience for users.

More info: https://zindi.africa/competitions/galsenais-fraud-detection-competition

---

## 🧠 Objective

Build a robust machine learning model to classify whether a given transaction is **fraudulent** or **legitimate** based on the available features.

---

## 📁 Dataset Description

The dataset includes:

* Approximately **140,000** transaction records
* Time window: **Nov 15, 2018 – Mar 15, 2019**
* Labeled entries indicating whether a transaction was **fraudulent**
* Various features related to transaction metadata and user behavior



## 🧪 Key Components

* 🔍 **EDA**: Analyzing class imbalance, transaction trends, and feature distributions
* 🧹 **Preprocessing**: Handling missing values, feature encoding, scaling
* 🧠 **Modeling**: Tree-based methods (e.g., XGBoost, Random Forest), Logistic Regression, and Neural Networks
* 🧾 **Evaluation**: F1 Score to focus on rare class (fraudulent transactions)
* 📉 **Imbalance Handling**: undersampling, class-weight tuning

---

## 🚀 Getting Started

### Prerequisites

* Python 3.12

```bash
pip install -e .
```

## 📈 Results

The best-performing model achieved:

* **F1-Score**: 74%


## 🤝 Acknowledgments

Thanks to **Xente** for providing the dataset and defining a high-impact problem in financial services.

