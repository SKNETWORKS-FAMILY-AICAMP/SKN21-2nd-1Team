import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from data_preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)

import sys
from pathlib import Path
# Add project root to path
project_root = Path.cwd()  # parent of notebooks/
sys.path.append(str(project_root))

from src.train_model import TrainModel
from util import params, utils

class MainModel:
    def __init__(self):
        print("=========START=========")

        self.processing = Preprocessing()

        self.train_model = TrainModel(params.models_with_params)

    def _predict(X_test, y_test):
        with open("../result/XGBoost.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        pred = xgb_model.predict(X_test)
        proba = xgb_model.predict_proba(X_test)

        print(pred, proba)

        return proba


    def main(self, df, only_predict = False, thresh = 0.5):
        # data preprocessing
        X = df.drop(columns=["churn"], errors="ignore")
        y = df['churn']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.25,
                                                            stratify=y,
                                                            random_state=42
                                                            )
        
        self.processing = self.processing.fit(X_train)
        X_train = self.processing.transform(X_train)

        X_test = self.processing.transform(X_test)
        
        if only_predict:
            with open("./result/XGBoost.pkl", "rb") as f:
                gbm_model = pickle.load(f)
            best_models = {"XGBoost":gbm_model}
        else: 
            # train ============================
            best_models = self.train_model.fit(X_train, y_train)
        
        train_results, score_results = self.train_model.predict(best_models, X_test, y_test, thresh)

        for name, _, proba in train_results:
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_fig, roc_ax = plt.subplots(figsize=(5, 3))
            roc_ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y_test, proba):.3f}")
            roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="무작위 기준")
            roc_ax.set_title(f"{name} ROC Curve")
            roc_ax.set_xlabel("False Positive Rate")
            roc_ax.set_ylabel("True Positive Rate")
            roc_ax.legend()
            plt.show()

        print(train_results)
        print(score_results)
        print("FINISH!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    df = pd.read_csv("./data/bank_customer_churn_prediction.csv")

    model = MainModel()
    model.main(df, only_predict=False, thresh = 0.2)