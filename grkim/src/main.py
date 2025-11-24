import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from data_preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
        self.processing.fit(df)
        new_df = self.processing.transform(df)

        X = new_df
        y = df['churn']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.25,
                                                            stratify=y,
                                                            random_state=42
                                                            )
        
        
        if only_predict:
            with open("./result/XGBoost.pkl", "rb") as f:
                xgb_model = pickle.load(f)
            best_models = {"XGBoost":xgb_model}
        else: 
            # train ============================
            best_models = self.train_model.fit(X_train, y_train)
        
        train_results, score_results = self.train_model.predict(best_models, X_test, y_test, thresh)

        for name, _, proba in train_results:
            plt.figure(figsize=(8,5))
            plt.hist(proba, bins=30, edgecolor='black')
            plt.title(f"Distribution of predict_proba - {name} (Churn Probability)")
            plt.xlabel("Predicted Probability of Churn")
            plt.ylabel("Count")
            plt.show()

        print(train_results)
        print(score_results)
        print("FINISH!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    df = pd.read_csv("./data/bank_customer_churn_prediction.csv")

    model = MainModel()
    model.main(df, only_predict=False, thresh = 0.2)