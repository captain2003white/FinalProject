import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import train_and_evaluate_model, preprocess_data

def test_data_loading():
    """测试数据加载功能"""
    # 这个测试需要实际数据文件，在CI中可能会跳过
    try:
        df = pd.read_csv('data/train_and_test2.csv')
        assert not df.empty
    except FileNotFoundError:
        pytest.skip("数据文件不存在，跳过测试")

def test_preprocess_data():
    """测试数据预处理"""
    # 创建测试数据
    test_data = pd.DataFrame({
        'Passengerid': [1, 2, 3],
        '2urvived': [0, 1, 0],
        'Age': [25, 30, 35],
        'Fare': [10.0, 20.0, 30.0],
        'Embarked': ['S', 'C', 'S'],
        'zero': [0, 0, 0]
    })
    
    X, y = preprocess_data(test_data)
    
    # 断言检查
    assert 'Passengerid' not in X.columns
    assert '2urvived' not in X.columns
    assert 'zero' not in X.columns
    assert len(X) == 3
    assert len(y) == 3

def test_model_training():
    """测试模型训练功能"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model, scaler = train_and_evaluate_model(X, y)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert scaler is not None

def test_model_evaluation():
    """测试模型评估功能"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model, scaler = train_and_evaluate_model(X, y)
    
    # 生成预测
    X_test, _ = make_classification(n_samples=20, n_features=5, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    assert y_pred is not None
    assert len(y_pred) == 20

if __name__ == "__main__":
    pytest.main([__file__, "-v"])