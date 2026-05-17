import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.num_cols = None
        
    def load_data(self):
        """Loads data from CSV."""
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"File not found at {self.file_path}")

    def clean_data(self):
        """Performs initial cleaning and type conversion."""
        # Drop ID
        if 'customerID' in self.df.columns:
            self.df.drop('customerID', axis=1, inplace=True)
            
        # TotalCharges to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'].fillna(0, inplace=True)
        
        # Binary encoding
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        
    def encode_and_scale(self):
        """Handles encoding and scaling."""
        # Identify categorical columns
        cat_cols = [c for c in self.df.columns if self.df[c].dtype == 'object']
        self.num_cols = [c for c in self.df.columns if c not in cat_cols and c != 'Churn']
        
        # Label Encoding for binary categoricals, OHE for others
        le = LabelEncoder()
        for col in cat_cols:
            if self.df[col].nunique() <= 2:
                self.df[col] = le.fit_transform(self.df[col])
            else:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                
        # Scaling
        self.scaler = StandardScaler()
        self.df[self.num_cols] = self.scaler.fit_transform(self.df[self.num_cols])

        
    def handle_imbalance(self):
        """Applies SMOTE."""
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        self.feature_names = X.columns.tolist()
        return X_res, y_res

    def select_features(self, X, y, k=10):
        """Feature selection using SelectKBest."""
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        cols = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in cols]
        
        # Keep only selected features
        self.feature_names = selected_features
        return pd.DataFrame(X_new, columns=selected_features)

    def run_pipeline(self, feature_selection=True, k=15):
        self.load_data()
        self.clean_data()
        self.encode_and_scale()
        X, y = self.handle_imbalance()
        
        if feature_selection:
            X = self.select_features(X, y, k=k)
            
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
