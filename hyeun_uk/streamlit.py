from re import escape
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(
    page_title="강현욱 - 최종 모델 리포트",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        .header {
            text-align: center;
        }
        .box {
            margin: 0 auto;
        }
    </style>
    <div class="header">
        <h1 style='margin: 0;'>은행 이탈고객 예측 모델 리포트</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Feature Engineering")
feature_df = pd.DataFrame(
    {
        "특성": [
            "연령별 신용점수(credit_to_age)",
            "연봉 대비 계좌 잔고(balance_to_salary)",
            "연령 가중치(risk_age_rank)",
            "연령별 계좌보유기간 (tenure_to_age)",
        ],
        "특성 생성 방법": ["신용 점수 / 연령", "잔고 / 연봉", "연령대 구간화", "계좌보유 기간 / 연령",],
        "비고": [
            "",
            "",
            "20대 이하, 20대, 30대, 40대, 50대, 60대 이상", "",
        ],
    },
)
st.dataframe(feature_df, hide_index=True)

st.divider()

st.subheader("Preprocessing")
# st.markdown(":orange-badge[XGBoost, LightGBM 모델 특성상 스케일링을 적용하지 않음]")
encoding_df = pd.DataFrame(
    {
        "특성": ["도시(country)", "성별(sex)", "연령 가중치(risk_age_rank)"],
        "인코딩 방법": ["원핫 인코딩", "원핫 인코딩", "원핫 인코딩"],
        "비고": ["", "남성을 1로 가중치를 부여 하고자 함", ""],
    }
)
st.dataframe(encoding_df, hide_index=True)
# st.markdown(
#     """
#         <h4 style="font-size: 1.4rem">Data Splitting Strategy</h4>
#         <p>KFold를 초기 시도했으나 데이터 양 부족으로 인한 성능 저하</br> 최종 Hold-out 데이터셋으로 학습 진행</p>
#     """,
#     unsafe_allow_html=True,
# )

st.divider()


st.subheader("Models")
st.markdown(
    """
        <ul>
            <li>앙상블(RandomForest + XGBoost) 두 모델의 평균치로 최종 예측</li>
            <li>Target Data(이탈/유지) 불균형을 잡기위해 class 가중치를 다르게 설정</li>
        </ul>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <style>
            .model_box {
                padding: 20px;
                border: 1px solid #666;
                border-radius: 20px;
                margin-bottom: 40px;
            }
        </style>
        <div class="model_box">
            <h4>LightGBM의 장점</h4>
            <ul>
                <li>빠른 학습 속도</li>
                <li>과적합 제어 기능 탑재</li>
                <li>결측치 자동 처리</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="model_box">
            <h4>XGBoost의 장점</h4>
            <ul>
                <li>과적합에 강함</li>
                <li>해석력이 좋음</li>
                <li>결측치, 이상치에 강함</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

parameter_df = pd.DataFrame(
    {
        "모델": [
            "RandomForest",
            "RandomForest",
            "RandomForest",
            "RandomForest",
            "RandomForest",
            "XGBoost",
            "XGBoost",
            "XGBoost",
            "XGBoost",
            "XGBoost",
            "XGBoost",
            "XGBoost",
            "XGBoost",
        ],
        "파라미터": [
            "random_state",
            "n_estimators",
            "max_depth",
            "n_jobs",
            "class_weight",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "eval_metric",
            "random_state",
            "tree_method",
        ],
        "사용한 값": [
            "42",
            "100",
            "10",
            "-1",
            "balanced",
            "400",
            "4",
            "0.05",
            "0.8",
            "0.8",
            "auc",
            "42",
            "hist",
        ]
    }
)

parameter_df["모델"] = parameter_df["모델"].mask(parameter_df["모델"].duplicated(), "")
st.dataframe(parameter_df, hide_index=True)

st.divider()

st.subheader("Post processing")
st.markdown(
    """
    <ul>
        <li>Recall 0.7이상 목표: 이탈 고객을 놓치지 않는 것이 중요</li>
        <li>Precision 0.5이상 목표: 너무 많은 고객을 이탈로 분류하지 않도록 조정</li>
        <li>최종 임계값 0.30 적용</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

threshold_comparison = pd.DataFrame(
    {
        "임계값": [0.50, 0.30],
        "Precision": [0.7366, 0.5498],
        "Recall": [0.4319, 0.6649],
        "F1-Score": [0.5446, 0.6019],
        "Accuracy": [0.8528, 0.8208],
    }
)

st.dataframe(threshold_comparison, hide_index=True)
st.divider()

st.subheader("ROC AUC Curve")

script_dir = Path(__file__).parent
y_test = pd.read_csv(script_dir / "data" / "y_test.csv")
ensemble_test_pred = pd.read_csv(script_dir / "data" / "xgb_test_proba.csv")
y_valid = pd.read_csv(script_dir / "data" / "y_valid.csv")
ensemble_valid_pred = pd.read_csv(script_dir / "data" / "xgb_valid_proba.csv")

fig, ax = plt.subplots(figsize=(8, 6))

v_rfc_roc = roc_auc_score(y_valid, ensemble_valid_pred)
v_fpr, v_tpr, _ = roc_curve(y_valid, ensemble_valid_pred)

t_rfc_roc = roc_auc_score(y_test, ensemble_test_pred)
t_fpr, t_tpr, _ = roc_curve(y_test, ensemble_test_pred)

ax.plot(v_fpr, v_tpr, label=f"Validation Set (AUC = {v_rfc_roc:.4f})")
ax.plot(t_fpr, t_tpr, label=f"Test Set (AUC = {t_rfc_roc:.4f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("최종 성능 요약")
final_metrics = pd.DataFrame(
    {
        "Metric": ["ROC AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [0.8671, 0.8004, 0.4987, 0.7706, 0.6055],
    }
)
st.dataframe(final_metrics, hide_index=True)

st.divider()
