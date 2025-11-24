import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             recall_score, precision_score, f1_score, accuracy_score,
                             PrecisionRecallDisplay, average_precision_score, precision_recall_curve,
                             RocCurveDisplay, roc_auc_score, roc_curve)

def plot_precision_recall_curve(y_proba, pred_proba, estimator_name=None, title=None):
    """Precision Recall Curve 시각화 함수
    Args:
        y_proba: ndarray - 정답
        pred_proba: 모델이 추정한 양성(Positive-1)일 확률
        estimator_name: str - 모델 이름. 시각화시 범례에 출력할 모델이름
        title: str - plot 제목
    Returns:
    Raises:"""
    # ap score 계산
    ap_score = average_precision_score(y_proba, pred_proba)
    # thresh 변화에 따른 precision, recall 값들 계산.
    precision, recall, _ = precision_recall_curve(y_proba, pred_proba)
    # 시각화
    disp = PrecisionRecallDisplay(
        precision, recall, 
        average_precision=ap_score,  
        estimator_name=estimator_name
    )
    disp.plot()
    if title:
        plt.title(title)
    plt.show()