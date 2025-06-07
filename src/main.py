import argparse
from src.utils.data_loader import load_data
from src.models.svm import SVM
from src.evaluation.metrics import evaluate_model
from src.visualization.plot import plot_results
import numpy as np
import pandas as pd

# 新增：支持sklearn的SVC用于非线性核
from sklearn.svm import SVC

def main():
    parser = argparse.ArgumentParser(description='SVM分类器，支持线性与非线性核')
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf'], help='SVM核函数类型')
    args = parser.parse_args()

    # 读取原始数据
    df = pd.read_csv('src/data/iris_for_svm.csv')
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # 训练集：非unknown，测试集：unknown
    is_unknown = labels == 'unknown'
    X_train, y_train = features[~is_unknown], labels[~is_unknown]
    X_test = features[is_unknown]

    # 标签转为二分类（setosa:-1, 其他:1）
    y_train_bin = np.where(y_train == 'setosa', -1, 1)

    if args.kernel == 'linear':
        # 线性SVM（自定义实现）
        svm_model = SVM(X_train, y_train_bin)
        initial_alpha = np.zeros(len(y_train_bin))
        svm_model.fit(initial_alpha)
        y_pred = svm_model.predict(X_train)
        # unknown预测
        if len(X_test) > 0:
            test_pred = svm_model.predict(X_test)
            print(f"unknown数据预测结果: {test_pred}")
        # 评估
        metrics = evaluate_model(y_train_bin, y_pred)
        print("训练集评估指标:", metrics)
        plot_results(X_train, y_train_bin, y_pred, svm_model, metrics)
    else:
        # 非线性SVM（RBF核，sklearn实现）
        clf = SVC(kernel='rbf', C=10.0)
        clf.fit(X_train, y_train_bin)
        y_pred = clf.predict(X_train)
        if len(X_test) > 0:
            test_pred = clf.predict(X_test)
            print(f"unknown数据预测结果: {test_pred}")
        metrics = evaluate_model(y_train_bin, y_pred)
        print("训练集评估指标:", metrics)
        # 画图（只支持2D特征）
        if X_train.shape[1] == 2:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_bin, edgecolors='k', marker='o')
            plt.title('SVM (RBF核) Decision Boundary')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

if __name__ == "__main__":
    main()