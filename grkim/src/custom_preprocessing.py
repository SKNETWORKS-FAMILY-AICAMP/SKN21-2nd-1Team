import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import LabelEncoder

class InteractionFeaturePreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_encoder = LabelEncoder()

        # age_group 설정용 bins, labels 저장
        self.age_bins = [0, 30, 40, 50, 60, 100]
        self.age_labels = ['20s', '30s', '40s', '50s', '60+']


    def fit(self, X, y = None):
        # age_group -> fit LabelEncoder
        age_group_series = pd.cut(
            X['age'],
            bins=self.age_bins,
            labels=self.age_labels
        )
        self.age_encoder.fit(age_group_series)
        return self


    def transform(self, X):
        df = X.copy()

        # 1) age_group 생성 (중복 없음)
        df['age_group'] = pd.cut(
            df['age'],
            bins=self.age_bins,
            labels=self.age_labels
        )
        # print(df['age_group'].head())
        # 2) age_group 인코딩
        df['age_group'] = self.age_encoder.transform(df['age_group'])

        # Age × Active Member
        df["age_x_active"] = df["age"] * df["active_member"]
        # Product × Active Member
        df["products_x_active"] = df["products_number"] * df["active_member"]
        # Age × Product Number
        df["age_x_products"] = df["age"] * df["products_number"]
        # estimated_salary × age
        df["estimated_x_age"] = df["estimated_salary"] * df["age"]
        # estimated_salary × active_member
        df["estimated_x_active"] = df["estimated_salary"] * df["active_member"]

        return df