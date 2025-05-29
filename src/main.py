from src.utils.data_loader import load_data
from src.models.svm import SVM
from src.evaluation.metrics import evaluate_model
from src.visualization.plot import plot_results
import numpy as np
import pandas as pd



def main():
    # Load the dataset
    data, labels = load_data('src/data/iris_for_svm.csv')

    # 排除最后一行（unknown测试项）
    train_data, test_data = data[:-1], data[-1:]
    train_labels, test_label = labels[:-1], labels[-1]

    # Initialize the SVM model
    svm_model = SVM(train_data, train_labels)

    # Fit the model
    initial_alpha = np.zeros(len(train_labels))
    svm_model.fit(initial_alpha)

    # Make predictions
    predictions = svm_model.predict(train_data)

    # Evaluate the model
    metrics = evaluate_model(train_labels, predictions)
    print("Evaluation Metrics:", metrics)

    # Visualize the results
    plot_results(train_data, train_labels, predictions, svm_model, metrics)

    # 对unknown测试项进行预测并输出
    test_pred = svm_model.predict(test_data)
    print(f"测试项预测label: {test_pred[0]}")

if __name__ == "__main__":
    main()