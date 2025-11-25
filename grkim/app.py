import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_preprocessing import Preprocessing

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="은행 고객 이탈률 예측", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = BASE_DIR / "data" / "bank_customer_churn_prediction.csv"
    return pd.read_csv(data_path)


@st.cache_resource(show_spinner=False)
def load_model():
    model_dir = BASE_DIR / "result"
    name = "LightGBM"
    model_path = model_dir / f"{name}.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return name, model
    return None, None


@st.cache_resource(show_spinner=False)
def build_preprocessor(df: pd.DataFrame) -> Preprocessing:
    preprocessor = Preprocessing()
    preprocessor.fit(df)
    return preprocessor


def evaluate_models(
    y_test,
    proba,
    model_name: str,
    threshold: float,
):
    rows = []
    preds = (proba >= threshold).astype(int)

    if y_test is not None:
        rows.append(
            {
                "모델": model_name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1": f1_score(y_test, preds),
                "ROC-AUC": roc_auc_score(y_test, proba),
                "AP": average_precision_score(y_test, proba),
            }
        )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(proba, bins=30, color="#1f77b4", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", label="임계값")
    ax.set_title(f"{model_name} 이탈 확률 분포")
    ax.set_xlabel("예측된 이탈 확률")
    ax.set_ylabel("고객 수")
    ax.legend()
    prob_fig = fig

    roc_fig = None
    if y_test is not None:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_fig, roc_ax = plt.subplots(figsize=(5, 3))
        roc_ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y_test, proba):.3f}")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="무작위 기준")
        roc_ax.set_title(f"{model_name} ROC Curve")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.legend()

    metrics_df = pd.DataFrame(rows).round(3) if rows else None
    return metrics_df, prob_fig, roc_fig


def get_default_inputs(df: pd.DataFrame) -> dict:
    defaults = {
        "credit_score": int(df["credit_score"].median()),
        "age": int(df["age"].median()),
        "tenure": int(df["tenure"].median()),
        "balance": float(df["balance"].median()),
        "products_number": int(df["products_number"].median()),
        "estimated_salary": float(df["estimated_salary"].median()),
        "country": df["country"].mode()[0],
        "gender": df["gender"].mode()[0],
    }
    return defaults


def predict_single(model, model_name: str, preprocessor: Preprocessing, user_df: pd.DataFrame, threshold: float):
    X_user = preprocessor.transform(user_df)
    prob = float(model.predict_proba(X_user)[0, 1])
    label = int(prob >= threshold)
    return {"model": model_name, "prob": prob, "label": label}


def churn_distribution_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = df["churn"].value_counts().sort_index()
    labels = ["유지(0)", "이탈(1)"]
    ax.bar(labels, counts, color=["#4daf4a", "#e41a1c"], edgecolor="black")
    ax.set_title("이탈/유지 분포")
    ax.set_ylabel("고객 수")
    return fig


def churn_by_country_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    churn_rate = df.groupby("country")["churn"].mean().sort_values(ascending=False)
    churn_rate.plot(kind="bar", color="#ff7f0e", edgecolor="black", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("이탈률")
    ax.set_title("국가별 평균 이탈률")
    return fig


def churn_by_age_chart(df: pd.DataFrame):
    # 연령별 이탈률을 세분화해 고령자의 이탈 상승 추세를 강조
    age_rate = df.groupby("age")["churn"].mean().sort_index()
    age_counts = df["age"].value_counts().sort_index()

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(age_rate.index, age_rate.values, color="#6baed6", alpha=0.8, label="연령별 이탈률")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("이탈률")
    ax1.set_xlabel("나이")

    ax2 = ax1.twinx()
    ax2.bar(age_counts.index, age_counts.values, color="#c7c7c7", alpha=0.35, width=0.8, label="연령 분포")
    ax2.set_ylabel("고객 수")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    ax1.set_title("연령별 이탈률과 고객 분포")
    return fig


def correlation_heatmap(df: pd.DataFrame):
    df_num = df.drop(columns=["customer_id", "country", "gender"], errors="ignore")
    corr = df_num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=False, cmap="Blues", ax=ax)
    ax.set_title("컬럼 상관관계 히트맵")
    return fig


def main():
    st.title("은행 고객 이탈률 예측 대시보드")
    st.caption("은행 고객의 이탈 가능성을 빠르게 확인하고, 모델 분포와 결과를 함께 확인하세요.")

    df = load_data()
    model_name, model = load_model()
    if model is None:
        st.error("result 폴더에 학습된 모델(.pkl)을 넣어주세요. (예: LightGBM.pkl)")
        return

    X = df.drop(columns=["churn"], errors="ignore")
    y = df["churn"] if "churn" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    threshold = st.session_state.get("threshold_value", 0.2)
    defaults = get_default_inputs(df)

    preprocessor = build_preprocessor(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    tab_overview, tab_predict, tab_data = st.tabs(["요약/결과", "예측하기", "데이터 살펴보기"])

    with tab_overview:
        st.subheader("모델 성능 및 분포")
        threshold = st.slider(
            "이탈 판정 기준(확률 default = 0.2)",
            0.1,
            0.9,
            float(threshold),
            0.05,
            key="threshold_value",
            help="임계값을 높이면 보수적으로 이탈을 예측하며, 낮추면 더 많은 이탈 위험을 포착합니다.",
        )
        st.caption(f"적용 중인 모델: {model_name}")
        st.info("성능 지표와 그래프는 저장된 `LightGBM.pkl` 모델을 그대로 불러와 계산합니다.  \n" \
                "**이탈 고객을 놓치지 않는 것**이 최우선이라 Recall을 약 0.8까지 높이는 데 집중했고,  \n" \
                "Precision은 예측 정확도를 고려해 0.5 아래로 떨어지지 않도록 설정했습니다.")

        metrics_df, prob_fig, roc_fig = evaluate_models(
            y_test, proba, model_name, threshold
        )
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("평가할 타깃(Churn) 컬럼이 없거나 계산할 수 없습니다.")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(prob_fig)
        with col2:
            if roc_fig:
                st.pyplot(roc_fig)

    with tab_predict:
        st.subheader("고객 정보 입력")
        countries = sorted(df["country"].unique())
        genders = sorted(df["gender"].unique())

        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            credit_score = c1.slider(
                "신용 점수", int(df["credit_score"].min()), int(df["credit_score"].max()), defaults["credit_score"]
            )
            age = c2.slider("나이", int(df["age"].min()), int(df["age"].max()), defaults["age"])

            tenure = c1.slider("거래 기간(년)", int(df["tenure"].min()), int(df["tenure"].max()), defaults["tenure"])
            balance = c2.number_input(
                "계좌 잔고",
                float(df["balance"].min()),
                float(df["balance"].max()),
                float(defaults["balance"]),
                step=100.0,
                format="%.2f",
            )

            products_number = c1.slider(
                "보유 상품 개수",
                int(df["products_number"].min()),
                int(df["products_number"].max()),
                defaults["products_number"],
            )
            estimated_salary = c2.number_input(
                "연봉(추정)",
                float(df["estimated_salary"].min()),
                float(df["estimated_salary"].max()),
                float(defaults["estimated_salary"]),
                step=100.0,
                format="%.2f",
            )

            country = c1.selectbox("국가", countries, index=countries.index(defaults["country"]))
            gender = c2.selectbox("성별", genders, index=genders.index(defaults["gender"]))
            credit_card = c1.radio("신용카드 보유 여부", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오")
            active_member = c2.radio("활성 고객 여부", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오")

            submitted = st.form_submit_button("이탈 확률 예측")

        if submitted:
            print("submitted!!!")
            user_df = pd.DataFrame(
                [
                    {
                        "customer_id": 99999999,
                        "credit_score": credit_score,
                        "country": country,
                        "gender": gender,
                        "age": age,
                        "tenure": tenure,
                        "balance": balance,
                        "products_number": products_number,
                        "credit_card": int(credit_card),
                        "active_member": int(active_member),
                        "estimated_salary": estimated_salary,
                    }
                ]
            )

            result = predict_single(model, model_name, preprocessor, user_df, threshold)
            st.markdown("#### 예측 결과")
            prob_pct = result["prob"] * 100
            label_msg = "이탈 위험" if result["label"] else "유지 가능"
            st.metric(
                f"{result['model']} 확률 (임계값 : {threshold})",
                f"{prob_pct:.1f}%",
                label_msg,
                delta_color="inverse" if result["label"] else "normal",
            )

    with tab_data:
        st.subheader("데이터 탐색")
        st.markdown(
            "데이터는 Kaggle Dataset을 참고하여 [Bank Customer Churn Dataset]"
            "(https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) 기준으로 로딩됩니다."
        )
        st.write("상위 5개 행 미리보기")
        st.dataframe(df.head(), use_container_width=True)
        st.divider()
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.pyplot(churn_distribution_chart(df))
        with col2:
            st.pyplot(churn_by_age_chart(df))
        st.divider()
        st.subheader("컬럼 상관관계")
        st.pyplot(correlation_heatmap(df))
        st.info(
            "원본 Feature 간 상관관계가 낮아,  \n"
            "데이터 전처리 단계에서 의미 있는 칼럼들(각 모델들의 feature_importances)의 조합으로  \n"
            "상호작용 피처를 추가해 모델이 관계를 더 잘 포착하도록 했습니다."
        )
        st.subheader("Feature Importance 요약")
        fi_table = pd.DataFrame(
            [
                ["LogisticRegression", "✅ active_member, gender_Male, country_France, country_Spain, age_group"],
                ["RandomForest", "✅ age, products_number, age_group, balance, active_member"],
                ["XGBoost", "age_group, ✅ products_number, active_member, age, country_Germany"],
                ["LightGBM", "✅ estimated_salary, credit_score, balance, age, tenure"],
            ],
            columns=["모델", "중요 피처 (상위 5개)"],
        )
        st.table(fi_table)
        st.subheader("추가한 상호작용 피처")
        interaction_table = pd.DataFrame(
            [
                ["age_x_active", "나이 × 활성 고객 여부"],
                ["products_x_active", "상품 개수 × 활성 고객 여부"],
                ["age_x_products", "나이 × 상품 개수"],
                ["estimated_x_age", "예상 연봉 × 나이"],
                ["estimated_x_active", "예상 연봉 × 활성 고객 여부"],
            ],
            columns=["상호작용 피처", "설명"],
        )
        st.table(interaction_table)


if __name__ == "__main__":
    main()
