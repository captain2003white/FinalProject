# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import warnings
import os
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
        df['Embarked'].fillna(method='ffill', inplace=True)
    
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
    svm = SVC(C=1, gamma=0.1, kernel='rbf')
    svm.fit(X_train_std, y_train)
    
    return svm, std

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

def main():
    """主函数"""
    # 从环境变量读取配置
    secret_key = os.getenv('MY_SECRET_KEY', 'default_secret')
    data_path = os.getenv('DATA_PATH', 'data/train_and_test2.csv')
    model_name = os.getenv('MODEL_NAME', 'SVM')
    
    print(f"Using secret key: {secret_key}")
    print(f"Data path: {data_path}")
    print(f"Model name: {model_name}")
    
    # 读取数据集
    df = pd.read_csv('data/train_and_test2.csv', encoding='utf-8')
    
    # 预处理数据
    X, y = preprocess_data(df)
    
    # 训练模型
    model, scaler = train_and_evaluate_model(X, y)
    
    # 划分测试集用于评估
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 评估模型
    accuracy, recall, precision, f1 = evaluate_model(model, scaler, X_test, y_test)
    
    return accuracy, recall, precision, f1

if __name__ == "__main__":
    main()