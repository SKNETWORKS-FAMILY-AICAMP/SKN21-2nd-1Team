import streamlit as st
# 제목
st.title("📌 고객 이탈 예측 모델 보고서 (Churn Prediction)")

st.markdown("---")

# 1. 프로젝트 개요
st.subheader("1. 프로젝트 개요")
st.markdown("""
본 프로젝트는 은행 고객 데이터를 기반으로 **이탈 여부(Churn)** 를 예측하는 모델을 구축하는 것을 목표로 한다.  
이탈 가능성이 높은 고객을 조기에 식별하여, 사전 대응 전략을 지원하는 시스템 구축이 핵심 목적이다.

전체 과정은  
**데이터 전처리 → 스케일링 → 모델 학습 → Soft Voting → Threshold 최적화 → AUC 평가**  
흐름으로 진행되었다.
""")

st.markdown("---")

# 2. 데이터 전처리
st.subheader("2. 데이터 전처리")
st.markdown("""
### 🔹 (1) 결측치 처리
- 본 데이터셋은 결측치가 존재하지 않아 별도 처리 미실시

### 🔹 (2) 인코딩
- Country, Gender 등 **범주형 변수에 One-Hot Encoding 적용**

### 🔹 (3) 스케일링
- 거리 기반 모델 및 안정적 학습을 위해 **StandardScaler** 적용  
- CreditScore, Balance, Age 등의 수치 변수 정규화

### 🔹 (4) Feature Engineering
- 실험 중 모델 성능 향상을 위해 다양한 파생 변수 구성  
  - Balance Ratio = Balance / EstimatedSalary  
  - Credit Usage Rate = Balance / CreditScore  
  - Tenure 그룹화  
- Soft Voting 조합에서 안정적 영향을 주는 변수 중심으로 선택

""")

st.markdown("---")

# 3. 모델 선정 배경
st.subheader("3. 모델 선정 배경")
st.markdown("""
최종 모델은 **Soft Voting Classifier**를 사용하였다.

### 🔹 선정 이유
- 단일 모델 성능의 편차를 줄이고 **안정적 예측** 확보
- Logistic Regression(해석 용이), Random Forest(비선형 대응), XGBoost(강력한 예측력)의 장점을 결합
- 분류 불균형 상황에서 상대적으로 **더 높은 Recall** 확보

### 🔹 Soft Voting 방식
- 각 모델의 **예측 확률(Soft Probability)** 을 평균  
- 더 부드럽고 안정적인 예측을 제공
""")

st.markdown("---")

# 4. 성능 향상을 위한 튜닝
st.subheader("4. 성능 향상을 위한 하이퍼파라미터 및 Threshold 최적화")
st.markdown("""
### 🔹 (1) Stratified K-Fold 기반 GridSearch
- 클래스 불균형을 고려하여 Stratified 방식 적용  
- 주요 튜닝 파라미터  
  - RandomForest: `n_estimators`, `max_depth`  
  - XGBoost: `learning_rate`, `max_depth`, `min_child_weight`  
  - Logistic Regression: `C`, `class_weight`  

### 🔹 (2) 불균형 데이터 대응 전략
- `class_weight='balanced'` 적용  
- Positive(이탈 고객) 비율이 낮아 **Recall** 개선을 최우선 목표로 설정

### 🔹 (3) 최적 Threshold 탐색 (너가 직접 만든 코드 기반)
- precision 최소 기준을 충족하는 threshold 중  
  **Recall을 최대화하는 지점을 자동 탐색**
- 0.50 ~ 0.70 구간에서 0.01 단위로 검색
- 최종 Soft Voting threshold 예: **약 0.58** (예시)

""")

st.markdown("---")

# 5. 최종 성능 (AUC-ROC 중심)
st.subheader("5. 최종 성능 평가")
st.markdown("""
### 🔹 AUC-ROC
- 최종 AUC: **약 0.86**  
  → 이탈/비이탈 고객 분리 성능이 우수함을 의미

### 🔹 Precision / Recall 균형
- Recall을 중점적으로 끌어올리되  
  Precision이 일정 수준 이하로 떨어지지 않도록 threshold 조정

### 🔹 Confusion Matrix 기반 검증
- Positive(이탈 고객) 클래스 탐지 비율 개선  
- Soft Voting 구조와 threshold 재조정으로 recall 향상 성공

""")

st.markdown("---")
