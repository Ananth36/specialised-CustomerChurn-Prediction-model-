import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, accuracy_score
import shap

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_proba = model.predict(X_test)
        self.y_pred = (self.y_pred_proba > 0.5).astype(int)
        
    def get_metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        return {
            "Accuracy": acc,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        }
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        return fig

    def explain_model(self):
        try:
            # TreeExplainer for LightGBM
            # check_additivity=False is crucial for newer LGBM versions to avoid common errors
            explainer = shap.TreeExplainer(self.model.model)
            shap_values = explainer.shap_values(self.X_test, check_additivity=False)
            
            # Handling different return types from shap_values
            # Binary classification might return list [class0, class1] or just class1 depending on version
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1] # Positive class
                else:
                     # Fallback if unexpected list structure
                    shap_values = shap_values[-1]
            
            return explainer, shap_values
        except Exception as e:
            # Return None to handle gracefully in UI
            print(f"SHAP Error: {e}")
            return None, None
