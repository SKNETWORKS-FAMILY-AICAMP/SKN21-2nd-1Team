from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class MultiModelTuner:
    def __init__(self, models_with_params, scoring="roc_auc", n_iter=20):
        """
        models_with_params = {
            "RandomForest": (RandomForestClassifier(), param_rf),
            "XGBoost": (XGBClassifier(), param_xgb),
            "LightGBM": (LGBMClassifier(), param_lgb),
        }
        """
        self.models_with_params = models_with_params
        self.scoring = scoring
        self.n_iter = n_iter
        self.best_models = {}

    def tune(self, X, y):
        X = X.copy()
        y = y.copy()
        for name, (model, param_space) in self.models_with_params.items():
            print(f"▶ Tuning {name} ...")

            # search = GridSearchCV(
            #     estimator=model,
            #     param_grid=param_space,
            #     cv=3,                       # 교차검증을 위한 fold 횟수
            #     return_train_score=True,
            #     refit=True,                 # True : 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킨다. (Default = True)
            #     n_jobs=-1,
            #     scoring=self.scoring,
            # )
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_space,
                scoring=self.scoring,
                n_iter=self.n_iter,
                cv=3,
                n_jobs=-1,
                random_state=42
            )

            search.fit(X, y)
            self.best_models[name] = search.best_estimator_
            print(f"✔ {name} tuning completed. Best model found.")
            print(f"  → {search.best_params_}\n")

        return self.best_models
