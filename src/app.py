# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import warnings
import os
import joblib
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv()

def preprocess_data(df):
    """数据预处理函数"""
    # 删除zero列
    for i in range(0, 19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        if col_name in df.columns:
            df.drop([col_name], inplace=True, axis=1)
    
    # 对缺失值进行向前填充
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].ffill()
    
    # 建立标签
    X = df.drop(['Passengerid', '2urvived'], axis=1)
    y = df['2urvived']
    
    return X, y

def train_and_evaluate_model(X, y, test_size=0.3, random_state=42):
    """训练和评估模型"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 数据标准化
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    
    # 实例化SVM
    svm = SVC(C=1, gamma=0.1, kernel='rbf', probability=True)
    svm.fit(X_train_std, y_train)
    
    return svm, std, X_test, y_test

def evaluate_model(model, scaler, X_test, y_test):
    """评估模型性能"""
    X_test_std = scaler.transform(X_test)
    y_pred = model.predict(X_test_std)
    
    # 计算混淆矩阵
    confusion = confusion_matrix(y_test, y_pred)
    print("原始混淆矩阵：")
    print(confusion)
    
    # 归一化混淆矩阵
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    print("\n归一化混淆矩阵：")
    print(confusion_normalized)
    
    # 计算模型评价指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'\naccuracy={accuracy:.4f}, recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}')
    
    # 打印详细的分类报告
    from sklearn.metrics import classification_report
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred, target_names=['负类', '正类']))
    
    return accuracy, recall, precision, f1

def save_model(model, scaler, X_columns, file_path="model"):
    """保存模型和预处理对象到pkl文件"""
    # 创建模型目录
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # 创建完整的模型状态
    model_state = {
        'model': model,
        'scaler': scaler,
        'feature_names': X_columns.tolist() if hasattr(X_columns, 'tolist') else list(X_columns),
        'model_type': 'SVM',
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存模型状态
    model_file = f"{file_path}_state.pkl"
    joblib.dump(model_state, model_file)
    print(f"模型状态已保存到: {model_file}")
    
    # 也可以单独保存模型和标准化器（可选）
    joblib.dump(model, f"{file_path}.pkl")
    joblib.dump(scaler, f"{file_path}_scaler.pkl")
    
    return model_file

def load_model(file_path):
    """从pkl文件加载模型状态"""
    model_state = joblib.load(file_path)
    print(f"模型加载成功，创建时间: {model_state.get('created_time', '未知')}")
    return model_state

def predict_with_model(model_state, new_data):
    """使用加载的模型进行预测"""
    model = model_state['model']
    scaler = model_state['scaler']
    
    # 确保输入数据的特征顺序正确
    if 'feature_names' in model_state:
        expected_features = model_state['feature_names']
        if hasattr(new_data, 'columns'):
            # 如果new_data是DataFrame，确保列顺序一致
            new_data = new_data[expected_features]
    
    # 标准化数据
    new_data_std = scaler.transform(new_data)
    
    # 进行预测
    predictions = model.predict(new_data_std)
    probabilities = model.predict_proba(new_data_std)
    
    return predictions, probabilities

def main():
    """主函数"""
    # 从环境变量读取配置
    secret_key = os.getenv('MY_SECRET_KEY', 'default_secret')
    data_path = os.getenv('DATA_PATH', 'data/train_and_test2.csv')
    
    print(f"Using secret key: {secret_key}")
    print(f"Data path: {data_path}")
    
    # 读取数据集
    df = pd.read_csv(data_path, encoding='utf-8')
    
    # 预处理数据
    X, y = preprocess_data(df)
    
    # 训练模型
    print("Training model...")
    model, scaler, X_test, y_test = train_and_evaluate_model(X, y)
    
    # 评估模型
    print("Evaluating model...")
    accuracy, recall, precision, f1 = evaluate_model(model, scaler, X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 保存模型
    print("\nSaving model...")
    model_file = save_model(model, scaler, X.columns, "models/svm_model")
    
    # 演示加载和预测
    print("\nTesting model loading and prediction...")
    loaded_model_state = load_model(model_file)
    
    # 使用测试集的一部分进行预测演示
    sample_data = X_test.head(5)
    predictions, probabilities = predict_with_model(loaded_model_state, sample_data)
    
    print(f"\n预测演示 (前5个样本):")
    print(f"预测结果: {predictions}")
    print(f"实际标签: {y_test.head(5).values}")
    print(f"预测概率: {probabilities}")
    
    return accuracy, recall, precision, f1, model_file

if __name__ == "__main__":
    accuracy, recall, precision, f1, model_file = main()