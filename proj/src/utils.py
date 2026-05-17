import pandas as pd
import numpy as np
import random

def generate_synthetic_data(filepath, n_rows=1000):
    """Generates a synthetic Telco Churn dataset."""
    print(f"Generating synthetic data at {filepath}...")
    
   
    tenure_dist = np.concatenate([np.random.randint(1, 12, int(n_rows*0.4)), np.random.randint(12, 72, int(n_rows*0.6))])
    np.random.shuffle(tenure_dist)
    
    # Contract tied to tenure
    contracts = []
    for t in tenure_dist:
        if t < 12: contracts.append(np.random.choice(['Month-to-month', 'One year'], p=[0.9, 0.1]))
        elif t < 24: contracts.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.5, 0.4, 0.1]))
        else: contracts.append(np.random.choice(['Two year', 'One year', 'Month-to-month'], p=[0.7, 0.2, 0.1]))
        
    data = {
        'customerID': [f'{i}A-B' for i in range(n_rows)],
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'SeniorCitizen': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_rows),
        'Dependents': np.random.choice(['Yes', 'No'], n_rows),
        'tenure': tenure_dist,
        'PhoneService': np.random.choice(['Yes', 'No'], n_rows),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'Contract': contracts,
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_rows),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_rows),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_rows),
    }
    
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] + np.random.normal(0, 10, n_rows)
    df['TotalCharges'] = df['TotalCharges'].abs().astype(str) # Convert to string for cleaning test
    
    # Generate Churn based on Logic (High Churn for: Low Tenure, Month-to-month, Fiber, Electronic Check)
    def failure_logic(row):
        score = 0
        if row['Contract'] == 'Month-to-month': score += 0.4
        if row['tenure'] < 12: score += 0.3
        if row['InternetService'] == 'Fiber optic': score += 0.2
        if row['PaymentMethod'] == 'Electronic check': score += 0.2
        if row['MonthlyCharges'] > 80: score += 0.1
        
        # Random noise
        prob = min(max(score + np.random.normal(0, 0.2), 0), 1)
        return 'Yes' if prob > 0.55 else 'No'

    df['Churn'] = df.apply(failure_logic, axis=1)
    
    # Introduce some blank TotalCharges for tenure=0 to test cleaning
    mask = df['tenure'] == 0
    df.loc[mask, 'TotalCharges'] = ' '
    
    df.to_csv(filepath, index=False)
    print("Synthetic data generated.")

if __name__ == "__main__":
    generate_synthetic_data('data/Telco-Customer-Churn.csv')
