import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
from pathlib import Path
# Add project root to path
project_root = Path.cwd()  # parent of notebooks/
sys.path.append(str(project_root))

from src.custom_preprocessing import InteractionFeaturePreProcessing

# Pipeline Version =====================================
class PipelinePreprocessing():
    def __init__(self):
        self.numeric_cols = ["age", "tenure", "credit_score", "estimated_salary", "products_number"]
        self.balance_col  = ["balance"] # 비 정규적인 데이터
        self.label_cols   = ["gender", "country"]
        self.drop_col     = ['customer_id']

    def create_pipeline(self, X_columns) -> Pipeline:
        transformer = ColumnTransformer([
            ('id_dropper', 'drop', [c for c in self.drop_col if c in X_columns]),
            ("standard_scaler", StandardScaler(), self.numeric_cols),
            ("robust_scaler", RobustScaler(), self.balance_col),
            ("oh_encoder", OneHotEncoder(sparse_output=False), self.label_cols)
        ], remainder='passthrough')

        pipeline = Pipeline([
            ("interaction_preprocessor", InteractionFeaturePreProcessing()),
            ("transformer", transformer)
        ], verbose=True)

        return pipeline


# =====================================
class Preprocessing:
    def __init__(self):
        # 0/1 binary
        self.binary_cols  = ["credit_card", "active_member"]
        
        # numeric cols
        self.numeric_cols = ["age", "tenure", "credit_score", "estimated_salary", "products_number"]

        # balance only
        self.balance_col  = ["balance"]

        # categorical OneHot 대상
        self.label_cols   = ["gender", "country"]
        
        self.standard_scaler = StandardScaler()
        self.robust_scaler  = RobustScaler() # or MinMaxScaler, log1p 변환 후 StandardScaler - balance
        self.onehot_encoder = OneHotEncoder(sparse_output=False) # sparse=False 반환타입 : Numpy array로 받기
        self.age_encoder    = LabelEncoder()

        # age_group 설정용 bins, labels 저장
        self.age_bins = [0, 30, 40, 50, 60, 100]
        self.age_labels = ['20s', '30s', '40s', '50s', '60+']


    def fit(self, df):
        # 1) 스케일러/인코더 학습
        self.standard_scaler.fit(df[self.numeric_cols])
        self.robust_scaler.fit(df[self.balance_col])

        # 2) age_group -> fit LabelEncoder
        age_group_series = pd.cut(
            df['age'],
            bins=self.age_bins,
            labels=self.age_labels
        )
        self.age_encoder.fit(age_group_series)
        
        # 3) OneHotEncoder 학습
        self.onehot_encoder.fit(df[self.label_cols])


    def transform(self, df):
        df = df.copy()

        # ID 제거
        df = df.drop(columns=['customer_id'], errors='ignore')

        # 1) age_group 생성 (중복 없음)
        df['age_group'] = pd.cut(
            df['age'],
            bins=self.age_bins,
            labels=self.age_labels
        )

        # 2) age_group 인코딩
        df['age_group'] = self.age_encoder.transform(df['age_group'])

        df[self.numeric_cols] = self.standard_scaler.transform(df[self.numeric_cols])
        df[self.balance_col]  = self.robust_scaler.transform(df[self.balance_col])

        # OneHot Encoder
        ohe_array = self.onehot_encoder.transform(df[self.label_cols])
        ohe_cols  = self.onehot_encoder.get_feature_names_out(self.label_cols)
        df_ohe    = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)

        # Interaction Feature
        # Age × Active Member
        df["age_x_active"] = df["age"] * df["active_member"]
        # # Product × Active Member
        df["products_x_active"] = df["products_number"] * df["active_member"]
        # # Age × Product Number
        df["age_x_products"] = df["age"] * df["products_number"]
        # # estimated_salary × age
        df["estimated_x_age"] = df["estimated_salary"] * df["age"]
        # # estimated_salary × active_member
        df["estimated_x_active"] = df["estimated_salary"] * df["active_member"]

        new_numeric_cols = self.numeric_cols.copy()
        new_numeric_cols.remove("age")

        # E. 최종 데이터 구성
        final_df = pd.concat([
            df[new_numeric_cols],
            df[self.balance_col],
            df[self.binary_cols],
            df_ohe,
            df[["age_group", "age_x_active", "products_x_active", "age_x_products", "estimated_x_age", "estimated_x_active"]]
        ], axis=1)

        print(">>>>> final columns : ", final_df.columns)

        return final_df


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
