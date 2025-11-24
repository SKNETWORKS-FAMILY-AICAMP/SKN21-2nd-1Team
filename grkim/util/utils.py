from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def get_score_list(y_test, pred, proba = None):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    return [accuracy, precision, recall, f1, auc, ap]
