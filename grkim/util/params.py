from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models_with_params = {
    # "LogisticRegression": (
    #     LogisticRegression(max_iter=3000, random_state=42),
    #     {
    #         "C": [0.001, 0.01, 0.1, 1, 3, 5, 10],   # 규제 강도
    #         "penalty": ["l1", "l2", "elasticnet"],
    #         "solver": ["saga"],
    #         "l1_ratio": [0, 0.3, 0.5, 0.7, 1]       # elasticnet 전용
    #     }
    # ),

    # "RandomForest": (
    #     RandomForestClassifier(random_state=42),
    #     {
    #         "n_estimators": [200, 300, 500],
    #         "max_depth": [4, 6, 8, 10],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4],
    #     }
    # ),

    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 7], # 최대 10인데 10을 넣으니까 Precision이 올라가고 Recall이 떨어짐. 극 일반화가 일어남.
            # "colsample_bytree": [0.6, 0.8, 1.0],
            # "scale_pos_weight": [4]        # 클래스 불균형일때, churn 데이터 -> 8:2
        }
    ),

    # LightGBM : leaf-wise라서 불균형 데이터에서 소수 클래스를 더 잘 포착하는 가지를 깊게 판다.
    "LightGBM": (
        LGBMClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "subsample": [0.6, 0.8, 1.0],
        }
    )

    # "XGBoost": (
    #     XGBClassifier(random_state=42),
    #     {
    #         # "n_estimators": [100, 200, 300],
    #         # "max_depth": [2, 3, 5],
    #         # "learning_rate": [0.05, 0.1]

    #         "n_estimators": [100, 200, 300, 500],   # (default = 100)
    #         "max_depth": [2, 3, 4, 5, 6],           # 양수만 가능 (default = 6)
    #         "learning_rate": [0.01, 0.05, 0.1],     # (default = 0.1)
    #         "subsample": [0.6, 0.8, 1.0],           # 각 트리마다 데이터 샘플링 비율 (default = 1) -> 데이터 개수 %
    #         "colsample_bytree": [0.6, 0.8, 1.0],    # 각 트리마다 feature 샘플링 비율 (default = 1) -> 60% feature만 볼건지, 80% feature만 볼건지, 100% 전체 feature볼건지
    #         "min_child_weight": [1, 3, 5, 7],       # 최소 child 노드 개수
    #         "gamma": [0, 0.1, 0.2, 0.5],            # split 최소 loss rkath
    #         "scale_pos_weight": [1, 2, 3, 4]        # 클래스 불균형일때, churn 데이터 -> 8:2
    #     }
    # ),

    # "LightGBM": (
    #     LGBMClassifier(random_state=42),
    #     {
    #         "n_estimators": [100, 200, 500],
    #         "num_leaves": [31, 63],
    #         "learning_rate": [0.05, 0.1]

    #         # "n_estimators": [100, 200, 300, 400, 500],   # boosting을 반복할 횟수 = 가중치 조정 횟수 (default = 100)
    #         # "learning_rate": [0.005, 0.01, 0.05, 0.1],   # default = 0.1 -> 0.05로 오름
    #         # "max_depth": [-1, 4, 6, 8, 15],              # default = -0.1
    #         # "num_leaves": [31, 60, 80],                     # default = 31
    #         # "max_bin": [63, 127, 255],                   # default = 255
    #         # "min_child_samples": [10, 20, 30],           # leaf 최소 데이터 수 (과적합 방지)
    #         # "subsample": [0.6, 0.8, 1.0],           # 각 트리마다 feature 샘플링 비율 (default = 1)
    #         # "colsample_bytree": [0.6, 0.8, 1.0],    # (default = 1)
    #      }
    # )
}
