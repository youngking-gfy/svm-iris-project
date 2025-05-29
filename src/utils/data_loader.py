def load_data(file_path):
    import pandas as pd
    import numpy as np

    try:
        data = pd.read_csv(file_path)
        if data.empty or len(data.columns) < 2:
            raise ValueError("数据文件为空或列数不足，请检查数据格式。")
        features = data.iloc[:, :-1].values
        labels = data.iloc[:, -1].values
        labels = np.where(labels == "setosa", -1, 1)  # Convert labels to -1 and 1
        return features, labels
    except FileNotFoundError:
        raise Exception(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        raise Exception(f"数据文件为空: {file_path}")
    except Exception as e:
        raise Exception(f"加载数据时发生错误: {e}")