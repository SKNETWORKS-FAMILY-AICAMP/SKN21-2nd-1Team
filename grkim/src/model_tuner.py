from sklearn.model_selection import RandomizedSearchCV

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
