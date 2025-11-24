from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from train_model import MultiModelTuner

class TrainModel:
    def __init__(self):
        self.models_with_params = {
            "LogisticRegression": (
                LogisticRegression(max_iter=3000, random_state=42),
                {
                    "C": [0.001, 0.01, 0.1, 1, 3, 5, 10],   # 규제 강도
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"],
                    "l1_ratio": [0, 0.3, 0.5, 0.7, 1]       # elasticnet 전용
                }
            ),

            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [200, 300, 500],
                    "max_depth": [4, 6, 8, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            ),

            "XGBoost": (
                XGBClassifier(eval_metric="logloss", random_state=42),
                {
                    "n_estimators": [200, 300, 500],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                }
            ),

            "LightGBM": (
                LGBMClassifier(random_state=42),
                {
                    "n_estimators": [200, 300, 500],
                    "max_depth": [-1, 4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                    "subsample": [0.6, 0.8, 1.0],
                }
            )
        }
        self.tuner = MultiModelTuner(self.models_with_params, n_iter=20)

        self.df_results = []

    
    def fit_transform(self, 
                      X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_test: pd.DataFrame, y_test: pd.DataFrame
                      ):
        best_models = self.tuner.tune(X_train, y_train)
        print(best_models)
        
        for name, model in best_models.items():            
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, proba)

            self.df_results.append([name, accuracy, precision, recall, f1, auc])

        for result in self.df_results:
            print(result)