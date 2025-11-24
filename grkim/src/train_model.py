import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix

from tuner_model import MultiModelTuner

import sys
from pathlib import Path
# Add project root to path
project_root = Path.cwd()  # parent of notebooks/
sys.path.append(str(project_root))

from util import utils

class TrainModel:
    def __init__(self, params):
        self.models_with_params = params
        self.tuner = MultiModelTuner(self.models_with_params, n_iter=20)

        self.score_results = []
        self.train_results = []

    def __save_models(self, models):
        save_dir = Path("./result/")
        # 폴더 없으면 자동 생성
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장
        for name, model in models.items():
            file_path = save_dir / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(model, f)

            print(f">>>> Saved to {file_path.resolve()}")

        print(">>>> Saved Finish!!!!")


    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # hyper parameter tuning
        best_models = self.tuner.tune(X_train, y_train)
        print(best_models)

        self.__save_models(best_models)

        return best_models  
    
    def predict(self, models, X_test, y_test, thresh):
        train_results = []
        score_results = []
        print(models)
        for name, model in models.items():            
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # 임계값 설정
            pred_thresh =  np.where(proba >= thresh, 1, 0)
            print(pred_thresh)

            confusion = confusion_matrix(y_test, pred_thresh)
            score_list = utils.get_score_list(y_test, pred_thresh, proba)

            print("혼동행렬(confusion) : \n", confusion, score_list)

            score_cols = ["Name", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "AP_SCORE"]
            score_df = pd.DataFrame([[name] + score_list], columns=score_cols)
            print("thresh : ", thresh, "\nScore DF : \n. ", score_df)

            train_results.append((name, pred_thresh, proba))
            score_results.append({name:score_df})

        return (train_results, score_results)