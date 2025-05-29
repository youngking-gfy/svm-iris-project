import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_decision_boundary(svm_model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def plot_performance_metrics(metrics):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['parameters'], metrics['accuracy'], label='Accuracy', marker='o')
    plt.plot(metrics['parameters'], metrics['precision'], label='Precision', marker='o')
    plt.plot(metrics['parameters'], metrics['recall'], label='Recall', marker='o')
    plt.plot(metrics['parameters'], metrics['f1_score'], label='F1 Score', marker='o')
    plt.title('Performance Metrics')
    plt.xlabel('Parameters')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

def plot_results(X, y_true, y_pred, svm_model, metrics=None):
    """
    综合可视化SVM模型结果，包括决策边界（仅限二维）、混淆矩阵和主要评估指标。
    """
    # 1. 决策边界（仅限二维特征）
    if X.shape[1] == 2:
        plot_decision_boundary(svm_model, X, y_true)
    else:
        print("特征维度大于2，跳过决策边界可视化。")

    # 2. 混淆矩阵
    plot_confusion_matrix(y_true, y_pred)

    # 3. 打印主要评估指标
    if metrics is not None:
        print("模型评估指标：")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        # 若未传入metrics，则简单计算并打印F1
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred)
        print(f"F1分数: {f1:.4f}")