import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class EliteCommanderModel:
    def __init__(self, params=None):
        self.model = None
        self.gamma = 2.0
        self.alpha = 0.25
        self.use_focal = True
        self.params = params if params else {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1, # Infinite depth (leaf-wise)
            'verbosity': -1
        }
    
    def focal_loss(self, preds, train_data):
        """
        Custom Focal Loss for LightGBM.
        Gamma: Focusing parameter
        Alpha: Balancing parameter
        """
        y_true = train_data.get_label()
        
        # Sigmoid
        sigmoid_p = 1.0 / (1.0 + np.exp(-preds))
        
        # Standard weighted cross entropy with focal components
        # pt = p if y=1 else 1-p
        pt = np.where(y_true == 1, sigmoid_p, 1 - sigmoid_p)
        
        # Focal term
        focal_weight = self.alpha * np.power(1 - pt, self.gamma)
        
        # Gradient scaling
        grad = focal_weight * (sigmoid_p - y_true)
        
        # Hessian scaling
        hess = focal_weight * sigmoid_p * (1 - sigmoid_p)
        
        return grad, hess

    def tune_focal_params(self, X_train, y_train, X_val, y_val):
        """
        Brute force tuning for gamma and alpha.
        """
        from sklearn.metrics import roc_auc_score
        
        gamma_range = [0.0, 1.0, 2.0, 3.0]
        alpha_range = [0.1, 0.25, 0.5, 0.75]
        
        best_auc = 0
        best_params = (2.0, 0.25)
        
        results = []
        
        for g in gamma_range:
            for a in alpha_range:
                self.gamma = g
                self.alpha = a
                
                # Training with short rounds for speed
                lgb_train = lgb.Dataset(X_train, y_train)
                
                # Create copy of params and set objective to focal_loss
                p = self.params.copy()
                p['objective'] = self.focal_loss
                
                temp_model = lgb.train(
                    p,
                    lgb_train,
                    num_boost_round=100
                )
                
                preds = temp_model.predict(X_val)
                # Convert logits to probabilities
                preds = 1.0 / (1.0 + np.exp(-preds))
                
                auc_score = roc_auc_score(y_val, preds)
                
                results.append({'gamma': g, 'alpha': a, 'auc': auc_score})
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_params = (g, a)
        
        self.gamma, self.alpha = best_params
        return results, best_params

    def train(self, X_train, y_train, X_val=None, y_val=None, use_focal=True):
        self.use_focal = use_focal
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train) if X_val is not None else None
        
        p = self.params.copy()
        p['is_unbalance'] = True 
        if use_focal:
            p['objective'] = self.focal_loss
        
        evals_result = {}
        self.model = lgb.train(
            p,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval] if lgb_eval else [lgb_train],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.record_evaluation(evals_result)
            ]
        )
        return evals_result

    def predict(self, X):
        preds = self.model.predict(X)
        if self.use_focal:
            # When using fobj, predict returns raw scores (logits)
            preds = 1.0 / (1.0 + np.exp(-preds))
        return preds
    
    def predict_binary(self, X, threshold=0.5):
        preds = self.predict(X)
        return (preds > threshold).astype(int)

    def get_feature_importance(self):
        return self.model.feature_importance()
