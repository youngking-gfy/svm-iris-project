# SVM 评估项目

本项目基于鸢尾花（Iris）数据集实现了支持向量机（SVM）分类模型，包含模型训练、性能评估及结果可视化等功能。

## 项目结构

```
svm-evaluation-project
├── src
│   ├── data
│   │   └── iris_for_svm.csv        # 用于训练和评估 SVM 模型的数据集
│   ├── models
│   │   └── svm.py                  # SVM 类的实现
│   ├── evaluation
│   │   └── metrics.py              # 模型性能评估函数
│   ├── visualization
│   │   └── plot.py                 # 结果可视化函数
│   ├── main.py                     # 项目入口
│   └── utils
│       └── data_loader.py          # 数据加载与预处理工具函数
├── requirements.txt                # 项目依赖
└── README.md                       # 项目文档
```

## 环境配置

1. 克隆仓库：
   ```
   git clone <repository-url>
   cd svm-evaluation-project
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

1. 准备数据集：
   确保 `iris_for_svm.csv` 文件位于 `src/data` 目录下。

2. 运行主程序：
   ```
   python -m src.main
   ```

程序将自动加载数据、训练 SVM 模型、评估性能并生成可视化结果。

## 评估指标

项目支持多种模型评估指标，包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1 分数（F1 Score）

## 可视化

可视化内容包括：
- 决策边界
- 混淆矩阵
- 不同参数设置下的性能指标

## 贡献

欢迎贡献！如有建议或改进，请提交 Pull Request 或 Issue。

## 许可证

本项目采用 MIT 许可证，详情见 LICENSE 文件。