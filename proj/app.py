import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap

from src.preprocessing import DataPipeline
from src.model import EliteCommanderModel
from src.evaluation import Evaluator
from src.utils import generate_synthetic_data

# Page Config
st.set_page_config(
    page_title="Telco Retention Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except:
    pass # Fallback if css missing

# Title
st.title("🛡️ Telco Retention Intelligence: AI Analytics Platform")
st.markdown("### Advanced Churn Risk Assessment Engine")

# Sidebar
st.sidebar.title("Analytics Control")
app_mode = st.sidebar.radio("Go to", ["1. Data Intelligence", "2. Model Training", "3. Strategic Evaluation", "4. Tactical Prediction"])

# Global State
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'evaluator' not in st.session_state:
    st.session_state['evaluator'] = None

# --- PHASE 1: DATA INTELLIGENCE ---
if app_mode == "1. Data Intelligence":
    st.header("Phase I: Data Preprocessing & Audit")
    
    # Smart Data Path Detection
    possible_paths = [
        'proj/WA_Fn-UseC_-Telco-Customer-Churn.csv', # The real user file
        'data/Telco-Customer-Churn.csv' # Fallback
    ]
    
    data_path = 'data/Telco-Customer-Churn.csv' # Default
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if not os.path.exists(data_path):
        st.warning("Data source missing. Initiating Synthetic Data Generator...")
        generate_synthetic_data(data_path)
        st.success("Synthetic Data Generated successfully.")
    else:
        st.info(f"Using Dataset: `{data_path}`")
    
    df_raw = pd.read_csv(data_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Assets")
        st.dataframe(df_raw.head())
        st.metric("Total Records", df_raw.shape[0])
        st.metric("Features", df_raw.shape[1])
        
    with col2:
        st.subheader("Initial Churn Distribution")
        fig_pie = px.pie(df_raw, names='Churn', title='Churn Imbalance', hole=0.4, color_discrete_sequence=['#00d2ff', '#ff4b4b'])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color':'white'})
        st.plotly_chart(fig_pie)

    st.markdown("---")
    st.subheader("Integrity Restoration & Processing")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        use_fs = st.checkbox("Enable Feature Selection", value=True)
    with col_ctrl2:
        k_features = st.slider("Number of Top Features", 5, 30, 15, disabled=not use_fs)

    if st.button("Execute Data Pipeline"):
        with st.spinner("Cleaning, Encoding, Scaling, and Applying SMOTE..."):
            pipeline = DataPipeline(data_path)
            X_train, X_test, y_train, y_test = pipeline.run_pipeline(feature_selection=use_fs, k=k_features)
            
            st.session_state['pipeline'] = pipeline
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.success("Pipeline Execution Complete.")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"Training Set Size: {X_train.shape[0]}")
            with col_b:
                st.info(f"Test Set Size: {X_test.shape[0]}")
                
            st.markdown("#### Selected Feature Space")
            st.dataframe(X_train.head())

# --- PHASE 2: MODEL TRAINING ---
elif app_mode == "2. Model Training":
    st.header("Phase II: Algorithmic Engine Design")
    
    if 'X_train' not in st.session_state:
        st.error("Data Pipeline not run. Please go to 'Data Intelligence' and execute pipeline.")
        st.stop()
        
    st.markdown("### LightGBM Architecture Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.05)
        num_leaves = st.slider("Num Leaves (Tree Complexity)", 10, 100, 31)
    with col2:
        if 'best_gamma' not in st.session_state:
            st.session_state['best_gamma'] = 2.0
        if 'best_alpha' not in st.session_state:
            st.session_state['best_alpha'] = 0.25
            
        cur_gamma = st.number_input("Focal Loss Gamma", 0.0, 5.0, st.session_state['best_gamma'])
        cur_alpha = st.number_input("Focal Loss Alpha", 0.1, 1.0, st.session_state['best_alpha'])

    if st.button("🚀 Tune Focal Hyperparameters"):
        with st.spinner("Searching for optimal Gamma/Alpha..."):
            temp_model = EliteCommanderModel(params={'objective': 'binary', 'metric': 'auc', 'learning_rate': learning_rate, 'num_leaves': num_leaves, 'verbose': -1})
            tune_results, best_params = temp_model.tune_focal_params(
                st.session_state['X_train'], st.session_state['y_train'],
                st.session_state['X_test'], st.session_state['y_test']
            )
            st.session_state['best_gamma'], st.session_state['best_alpha'] = best_params
            st.success(f"Best Parameters Found: Gamma={best_params[0]}, Alpha={best_params[1]}")
            
            # Show tuning heatmap or table
            tune_df = pd.DataFrame(tune_results)
            st.dataframe(tune_df.sort_values(by='auc', ascending=False))

    if st.button("Initialize & Train Elite Commander"):
        st.write("Training in progress...")
        progress_bar = st.progress(0)
        
        # Train
        model = EliteCommanderModel(params={
            'objective': 'binary', 
            'metric': 'auc', 
            'learning_rate': learning_rate, 
            'num_leaves': num_leaves,
            'verbose': -1
        })
        model.gamma = cur_gamma
        model.alpha = cur_alpha
        
        # Simulate progress for UX
        progress_bar.progress(50)
        
        history = model.train(st.session_state['X_train'], st.session_state['y_train'], st.session_state['X_test'], st.session_state['y_test'])
        st.session_state['model'] = model
        
        progress_bar.progress(100)
        st.balloons()
        st.success(f"Model Trained with Gamma={cur_gamma}, Alpha={cur_alpha}")
        
        # Show Basic initial metrics
        acc = history['valid_1']['auc'][-1]
        st.metric("Validation AUC", round(acc, 4))

# --- PHASE 3: EVALUATION ---
elif app_mode == "3. Strategic Evaluation":
    st.header("Phase III: Evaluation & Interpretation")
    
    if st.session_state['model'] is None:
        st.warning("Model not trained yet.")
        st.stop()
        
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    
    evaluator = Evaluator(model, X_test, y_test)
    evaluator.y_pred_proba = model.predict(X_test) # Refresh logic
    
    metrics = evaluator.get_metrics()
    
    # 1. Metrics Grid
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    c2.metric("ROC AUC", f"{metrics['ROC_AUC']:.4f}")
    c3.metric("PR AUC (Rare Event)", f"{metrics['PR_AUC']:.4f}", delta="Critical Metric")
    
    # 2. Confusion Matrix
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("The Error Matrix")
        fig_cm = evaluator.plot_confusion_matrix()
        st.pyplot(fig_cm)
        
    with col_viz2:
        st.subheader("Feature Importance")
        importances = model.get_feature_importance()
        feature_names = st.session_state['pipeline'].feature_names
        
        # Create DataFrame for Plotly
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='teal')
        fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color':'white'})
        st.plotly_chart(fig_feat)
        
    # 3. SHAP
    st.markdown("---")
    st.subheader("Global Interpretability (SHAP)")
    if st.button("Generate SHAP Analysis"):
        with st.spinner("Calculating SHAP values (this may take a moment)..."):
            explainer, shap_values = evaluator.explain_model()
            
            if shap_values is not None:
                st.write("Summary Plot")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig)
            else:
                st.error("Could not generate SHAP values. This might be due to a version mismatch or data issue.")

# --- PHASE 4: PREDICTION ---
elif app_mode == "4. Tactical Prediction":
    st.header("Tactical Customer Assessment")
    st.write("Simulate a customer profile to predict churn probability.")
    
    if st.session_state['pipeline'] is None:
        st.error("Pipeline not ready.")
        st.stop()
        
    # Expanded Input Form
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input("Tenure (Months)", 0, 72, 12)
    monthly_charges = c2.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
    total_charges = c3.number_input("Total Charges ($)", 0.0, 9000.0, 600.0)
    
    c4, c5, c6 = st.columns(3)
    contract = c4.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = c5.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = c6.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    # Advanced Settings within Expander
    with st.expander("⚙️ Advanced Prediction Settings", expanded=True):
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            threshold = st.slider("Decision Threshold (Sensitivity)", 0.0, 1.0, 0.35, 
                                help="Churn is a rare event. adjusting this threshold allows you to catch 'At Risk' customers earlier. A lower value (e.g., 0.35) makes the model more sensitive.")
        with col_t2:
            st.metric("Current Sensitivity", f"{1-threshold:.0%}")

    if st.button("Predict Churn Risk"):
        # Create input dataframe structure
        features = st.session_state['pipeline'].feature_names
        pipeline = st.session_state['pipeline']
        
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        
        # 1. Handle Categorical Inputs (One-Hot Mapping)
        # Function to safely set OHE bit
        def set_ohe(prefix, value):
            col_name = f"{prefix}_{value}"
            if col_name in features:
                input_data[col_name] = 1
                
        set_ohe("Contract", contract)
        set_ohe("InternetService", internet_service)
        set_ohe("PaymentMethod", payment_method)
        set_ohe("TechSupport", tech_support)
            
        # 2. Handle Numerics with Scaler
        if pipeline.scaler is not None and pipeline.num_cols is not None:
             nums = pd.DataFrame(np.zeros((1, len(pipeline.num_cols))), columns=pipeline.num_cols)
             
             if 'tenure' in pipeline.num_cols: nums['tenure'] = tenure
             if 'MonthlyCharges' in pipeline.num_cols: nums['MonthlyCharges'] = monthly_charges
             if 'TotalCharges' in pipeline.num_cols: nums['TotalCharges'] = total_charges
             
             # Fill mean for others (SeniorCitizen, etc - simplistic for demo)
             
             scaled_nums = pipeline.scaler.transform(nums)
             
             # Update input_data with scaled values
             for i, col in enumerate(pipeline.num_cols):
                 if col in features:
                     input_data[col] = scaled_nums[0][i]
        else:
            st.warning("Scaler not found. Running with raw values (Accuracy decreased).")
            # Fallback simple scaling
            if 'tenure' in features: input_data['tenure'] = (tenure - 32) / 24 
            if 'MonthlyCharges' in features: input_data['MonthlyCharges'] = (monthly_charges - 64) / 30


        # Prediction
        pred = st.session_state['model'].predict(input_data)
        prob = pred[0]
        
        # Display Results
        st.markdown("### Customer Risk Profile")
        
        gap = prob - threshold
        
        # Visual Guage
        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
             st.metric("Churn Probability", f"{prob:.1%}", delta=f"{gap*100:.1f}p vs Threshold", delta_color="inverse")
        
        with c_res2:
            # Dynamic Interpretation based on Slider
            if prob > (threshold + 0.25):
                st.error("🚨 **Critical Risk**: Highly Likely to Churn")
                st.write("Immediate retention offer recommended.")
            elif prob > threshold:
                st.warning("⚠️ **Moderate Risk**: Above Sensitivity Threshold")
                st.write("Customer shows early signs of departure.")
            elif prob > (threshold - 0.15):
                st.info("👀 **Watchlist**: Approaching Risk Threshold")
            else:
                st.success("🛡️ **Safe**: Loyal Customer")
                
            st.progress(min(prob, 1.0))
            st.caption(f"Risk Decision Boundary: {threshold:.0%}")
