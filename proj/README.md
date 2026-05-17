# Elite Commander: Telco Churn Prediction System

## Overview
A comprehensive Data Mining project designed to predict customer churn using an advanced **LightGBM** engine with **Focal Loss** for handling class imbalance. The system features a **Streamlit** frontend with premium "Glassmorphism" aesthetics.

## Phases Implemented
1. **Phase I: Data Preprocessing**
   - Structural Audit & Cleaning
   - Missing Value Treatment
   - One-Hot & Label Encoding
   - SMOTE (Synthetic Minority Over-sampling)

2. **Phase II: Model Making**
   - **Engine**: LightGBM (Leaf-wise growth)
   - **Optimization**: Focal Loss / Class Balancing for rare events
   - **Training**: Stratified Validation

3. **Phase III: Evaluation**
   - **metrics**: PR-AUC, ROC-AUC, Accuracy
   - **Visualization**: Confusion Matrix, Feature Importance
   - **Explanation**: SHAP (Global & Local)

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Explore**
   - Navigate through the Sidebar to "Data Intelligence" to clean the data.
   - Go to "Model Training" to build the Elite Commander model.
   - Go to "Strategic Evaluation" to view charts and SHAP values.
