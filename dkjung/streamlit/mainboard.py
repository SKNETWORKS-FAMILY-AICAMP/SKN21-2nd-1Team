import streamlit as st
from PIL import Image
import pathlib
import pandas as pd
import os

script_dir = pathlib.Path(__file__).parent
image_path = "script_dir/images/bank_churn.png"
COLS_PER_ROW = 3


def home_page():

    # st.image("./images/bank_churn.png", output_format="auto", use_column_width=None, clamp=False, channels="RGB")
    img = Image.open("./images/bank_churn.png")
    st.image(img, width=600)

    st.markdown("---")
    st.markdown("# 은행 고객 이탈 예측 분석")
    st.markdown("분석가: **정덕규**")

    st.markdown("---")
    st.markdown("### 개요")
    st.markdown("핀테크가 발전하고 다양한 은행 서비스가 증가함에 따라 은행 고객들의 이탈율이 증가함")
    st.markdown("은행 고객 이탈 분석과 예측을 위해 고객의 정보를 활용하여 **이탈율 감소**가 필요")

    st.markdown("---")
    st.markdown("### 프로젝트 목적")
    st.markdown("고객의 데이터를 이용해 인공지능 기술의 일환인 머신러닝을 활용하여 이탈율 분석 및 예측")
    st.markdown("다양한 고객의 정보들을 시각화하여 고객의 이탈율을 사전에 방지할 수 있도록 예측 **모델 구축**")

    st.markdown("---")
    st.markdown("### 고객 데이터 설명 및 전처리 방법")


    type_list = ['int', 'int', 'object', 'object', 'int', 'int', 'float', 'int', 'int', 'int', 'float', 'int']
    cat_list = ['customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number',
                'credit_card', 'active_member', 'estimated_salary', 'churn']
    description = ['계좌 ID', '신용점수', '국가', '성별', '나이', '계좌보유기간', '잔고', '상품개수', '신용카드 보유여부',
                   '활성 고객 여부', '예상 연봉', '이탈 여부']

    df = pd.DataFrame({
        'Type': type_list,
        'Category': cat_list,
        'Description': description
    })
    st.dataframe(df, width='stretch')

    st.info("머신러닝 모델에 학습하기 위해선 object type의 country와 gender를 전처리 필요함. "
                "Label Encoding으로 처리 후 데이터의 Training과 Validation set을 0.75: 0.25 비율로 나눔 (결측치는 존재하지 않음)")

    st.markdown("---")
    st.markdown("### 모델 및 하이퍼파라미터 선정")
    st.markdown("모델의 성능 평가를 위해 분류 모델 평가인 F1 score, Accuracy, ROC-AUC 점수 지표를 기준으로 아래와 같은 모델을 선정하였음")

    st.markdown("- Decision Tree")
    st.markdown("- Random Forest")
    st.markdown("- XGBoost")
    st.markdown("- LightGBM")
    st.markdown("- CatBoost")

    st.markdown("각 모델의 하이퍼파라미터 튜닝을 위해 Randomized search CV를 적용하였고 "
                "시계열 모델의 교차 검증을 위해 Time series cv를 5로 적용함")

    st.markdown("---")
    st.markdown("### 분석결과")

    df_read = pd.read_csv("../figure_results.csv")
    st.dataframe(df_read)
    st.markdown("ROC-AUC 기준으로 cat Boost가 제일 점수가 높으며 이를 검증하기 위해 이탈율과 고객의 데이터간 상관관계가 중요함")
    st.markdown("모델이 데이터를 학습하는데 있어서 예측에 중요한 변수들이 있으며 이에 대한 **Feature importance**는 아래와 같음")
    st.image(Image.open("./images/feature_importance.png"))
    st.markdown("주요 3개의 모델링(XGBoost, Light GBM, CatBoost) 중에 대다수 age, product_number였으며 XGBoost와 CatBoost는 각각 "
                "1위가 product_number, 2위가 age였고 LightGBM만 1위가 age로 판단됨. 즉 고객 이탈율에 있어서 **나이**가 우선순위가 높음")
    st.image(Image.open("./images/shap_value.png"))
    st.markdown("각 feature들이 얼마나 예측에 기여했는지 알아보기 위해 **Shap**(SHapley Addictive exPlanations) value를 추출하였으며 "
                "XGBoost와 LightGBM에서 1위가 **product_number**, 2위가 **age**의 기여도가 도출되었고 반대로 CatBoost는 그와 상반된 우선순위로 확인됨")
    st.image(Image.open("./images/roc-auc_curve.png"))
    st.markdown("5개의 모델에 대한 **ROC-AUC curve**는 상기 이미지와 같으며 Decision Tree의 기반 및 앙상블 모델인 XGBoost, CatBoost, LightGBM이"
                "높으며 그 중 **CatBoost**가 높은 성능을 보이고 있음")





    # 하단은 사이드바

    with st.sidebar:
        st.subheader("Main Page")
        st.selectbox(
            "Select Page",
            ("정덕규", "기타")
        )

        st.markdown("---")

        project_links = [
            {"label": "SKN21-2nd-1team", "url": "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN21-2nd-1Team"},
            {"label": "Bank Customer Churn Dataset",
             "url": "https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data"},
        ]

        st.sidebar.markdown("## Project links")
        for item in project_links:
            st.link_button(label=item["label"], url=item["url"], width="stretch")

        st.sidebar.markdown("## Library sources")
        library_links = [
            {"label": "Pandas", "url": "https://pandas.pydata.org/"},
            {"label": "Jupyter", "url": "https://jupyter.org/"},
            {"label": "Streamlit", "url": "https://streamlit.io/"},
            {"label": "Shap", "url": "https://shap.readthedocs.io/en/latest/"},
            {"label": "Scikit-learn", "url": "https://scikit-learn.org/stable/"},
            {"label": "Time series CV", "url": "https://scikit-learn.org/stable/module/cross_validation.html"},
            {"label": "Randomized search CV",
             "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html"},

        ]


        for i in range(0, len(library_links), COLS_PER_ROW):
            cols = st.columns(COLS_PER_ROW)
            for j in range(COLS_PER_ROW):
                item_index = i + j

                if item_index < len(library_links):
                    item = library_links[item_index]
                    with cols[j]:
                        st.link_button(
                            label=item["label"],
                            url=item["url"],
                            width='stretch'
                        )


        st.sidebar.markdown("## Modeling references")
        model_links = [
            {"label": "XGBoost", "url": "https://xgboost.readthedocs.io/en/stable/"},
            {"label": "CatBoost", "url": "https://catboost.ai/docs/en/"},
            {"label": "LightGBM", "url": "https://lightgbm.readthedocs.io/en/stable/"},
            {"label": "Decision Tree", "url": "https://scikit-learn.org/stable/modules/tree.html"},
            {"label": "Random Forest", "url": "https://www.ibm.com/kr-ko/think/topics/random-fores"},
            {"label": "Optuna", "url": "https://optuna.org/"},
        ]

        for i in range(0, len(model_links), COLS_PER_ROW):
            cols = st.columns(COLS_PER_ROW)
            for j in range(COLS_PER_ROW):
                item_index = i + j

                if item_index < len(model_links):
                    item = model_links[item_index]
                    with cols[j]:
                        st.link_button(
                            label=item["label"],
                            url=item["url"],
                            width='stretch'
                        )

    st.markdown("---")

home_page()