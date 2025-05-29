import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == -1) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def recall(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == -1))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }