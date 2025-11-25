# 모듈 불러오기
import os
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                             roc_curve)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost 설치가 안됐습니다... uv pip install xgboost 설치 바랍니다.") from exc

mpl.rcParams['font.family'] = "malgun Gothic"
mpl.rcParams['axes.unicode_minus'] = False





# 모델 성능 평가
def score_get(y_test, y_pred, proba=None):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)

    return [accuracy,
            precision,
            recall,
            f1,
            roc_auc]





print("imports 로드 완료")