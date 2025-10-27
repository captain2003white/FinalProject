# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import warnings
import os
import mlflow
import mlflow.sklearn
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

def train_with_mlflow(X, y, experiment_name="SVM_Experiment"):
    """使用MLflow追踪模型训练并推送到DagsHub"""
    
    # 配置DagsHub的MLflow跟踪URI
    mlflow.set_tracking_uri("https://dagshub.com/captain2003white/FinalProject.mlflow")
    
    # 设置DagsHub认证（如果需要）
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'captain2003white'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'bab328c8378dbe6d178cd5c02ad082b093ac2e97'
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"SVM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 记录参数
        mlflow.log_param("C", 1.0)
        mlflow.log_param("gamma", 0.1)
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)
        
        # 训练模型
        model, scaler = train_and_evaluate_model(X, y)
        
        # 评估模型
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test_std = scaler.transform(X_test)
        y_pred = model.predict(X_test_std)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        
        # 记录模型
        mlflow.sklearn.log_model(model, "svm_model")
        
        # 记录数据版本（通过DVC）
        mlflow.log_param("data_version", "v1.0")
        
        return model, scaler, accuracy, recall, precision, f1

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
    
    # 使用MLflow训练模型
    print("Training with MLflow tracking...")
    model, scaler, accuracy, recall, precision, f1 = train_with_mlflow(X, y)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, recall, precision, f1

if __name__ == "__main__":
    main()