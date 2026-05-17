# Project Plan: Telco Churn Prediction "Elite Commander"

## Overview
This project processes Telco customer data to predict churn using an advanced LightGBM model with Focal Loss. It includes a rich frontend for visualization and interaction.

## Phase I: Data Preprocessing (`src/preprocessing.py`)
- **Data Loading**: Load CSV (handle missing/dummy data).
- **Cleaning**: Fix "Total Charges", Label Encode binary cols, One-Hot Encode multi-class cols.
- **Scaling**: StandardScaler.
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique).
- **Feature Selection**: Correlation filter (>90%), remove IDs.

## Phase II: Model Making (`src/model.py`)
- **Engine**: LightGBM Classifier.
- **Optimization**: Custom Focal Loss (Gradient/Hessian).
- **Strategy**: Leaf-wise growth.
- **Training**: Stratified K-Fold.

## Phase III: Evaluation & Interpretation (`src/evaluation.py`)
- **Metrics**: PR-AUC, Confusion Matrix.
- **Comparisons**: vs Logistic Regression Baseline.
- **Interpretation**: SHAP values (Global & Local).

## Phase IV: Frontend (`app.py`)
- **Framework**: Streamlit.
- **Design**: Custom CSS for "Glassmorphism/Premium" feel.
- **Features**:
    - Data Explorer (View raw/processed data).
    - Model Training Dashboard (Control params, view training logs).
    - Performance Visualizer (Confusion Matrix, ROC/PR Curves).
    - "Why Churn?" Tool (SHAP Force plots for individual customers).

## Directory Structure
```
proj/
├── data/
│   └── Telco-Customer-Churn.csv (User provided or Synthetic)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── assets/
│   └── style.css
├── app.py
└── requirements.txt
```
