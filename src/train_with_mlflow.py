# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import numpy as np
import warnings
import os
import joblib
from datetime import datetime
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import subprocess
import git

warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv()

def get_git_commit_hash():
    """Get current git commit hash"""
    try:
        repo = git.Repo('.')
        return repo.head.commit.hexsha
    except:
        return "unknown"

def get_dvc_data_version(data_path):
    """Get DVC version hash for the data file"""
    try:
        # Get the DVC file path
        dvc_file = data_path + '.dvc'
        if os.path.exists(dvc_file):
            with open(dvc_file, 'r') as f:
                content = f.read()
                # Extract hash from DVC file
                for line in content.split('\n'):
                    if 'md5:' in line:
                        return line.split('md5:')[1].strip()
        return "unknown"
    except:
        return "unknown"

def preprocess_data_v1(df):
    """Data preprocessing function for v1 dataset"""
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

def preprocess_data_v2(df):
    """Data preprocessing function for v2 dataset (with additional features)"""
    # 删除zero列
    for i in range(0, 19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        if col_name in df.columns:
            df.drop([col_name], inplace=True, axis=1)
    
    # Handle categorical features
    if 'AgeGroup' in df.columns:
        le = LabelEncoder()
        df['AgeGroup_encoded'] = le.fit_transform(df['AgeGroup'].astype(str))
        df.drop(['AgeGroup'], axis=1, inplace=True)
    
    # 对缺失值进行向前填充
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(method='ffill', inplace=True)
    
    # 建立标签
    X = df.drop(['Passengerid', '2urvived'], axis=1)
    y = df['2urvived']
    
    return X, y

def train_and_evaluate_model(X, y, model_type='svm', test_size=0.3, random_state=42, hyperparams=None):
    """Train and evaluate model with hyperparameter tuning"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 数据标准化
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    
    if model_type == 'svm':
        if hyperparams:
            model = SVC(**hyperparams, probability=True)
        else:
            model = SVC(C=1, gamma=0.1, kernel='rbf', probability=True)
    elif model_type == 'rf':
        if hyperparams:
            model = RandomForestClassifier(**hyperparams)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train_std, y_train)
    
    return model, std, X_test, y_test, X_train, y_train

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance"""
    X_test_std = scaler.transform(X_test)
    y_pred = model.predict(X_test_std)
    
    # 计算混淆矩阵
    confusion = confusion_matrix(y_test, y_pred)
    
    # 计算模型评价指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, recall, precision, f1, y_pred, confusion

def hyperparameter_tuning(X_train, y_train, model_type='svm'):
    """Perform hyperparameter tuning"""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    
    if model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        model = SVC(probability=True)
    elif model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_std, y_train)
    
    return grid_search.best_params_, grid_search.best_score_

def run_experiment(data_path, dataset_version, model_type='svm', experiment_name=None, hyperparams=None):
    """Run a complete MLflow experiment"""
    
    # Set experiment name
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log metadata
        git_commit = get_git_commit_hash()
        data_version = get_dvc_data_version(data_path)
        
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("data_dvc_hash", data_version)
        
        # Log parameters
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)
        
        if hyperparams:
            for key, value in hyperparams.items():
                mlflow.log_param(f"hyperparam_{key}", value)
        
        # Read and preprocess data
        df = pd.read_csv(data_path, encoding='utf-8')
        
        if dataset_version == 'v1':
            X, y = preprocess_data_v1(df)
        else:  # v2
            X, y = preprocess_data_v2(df)
        
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("class_distribution", y.value_counts().to_dict())
        
        # Train model
        model, scaler, X_test, y_test, X_train, y_train = train_and_evaluate_model(
            X, y, model_type=model_type, hyperparams=hyperparams
        )
        
        # Evaluate model
        accuracy, recall, precision, f1, y_pred, confusion = evaluate_model(
            model, scaler, X_test, y_test
        )
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        
        # Log artifacts
        # Save model
        model_path = f"models/{model_type}_{dataset_version}_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Save scaler
        scaler_path = f"models/{model_type}_{dataset_version}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        # Save confusion matrix
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_type.upper()} on {dataset_version}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Not Survived', 'Survived'])
        plt.yticks(tick_marks, ['Not Survived', 'Survived'])
        
        # Add text annotations
        thresh = confusion.max() / 2.
        for i, j in np.ndindex(confusion.shape):
            plt.text(j, i, format(confusion[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        confusion_matrix_path = f"confusion_matrix_{model_type}_{dataset_version}.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, f"{model_type}_model")
        
        print(f"\nExperiment completed:")
        print(f"Model: {model_type}")
        print(f"Dataset: {dataset_version}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Git Commit: {git_commit}")
        print(f"Data Version: {data_version}")
        
        return accuracy, recall, precision, f1

def main():
    """Main function to run experiments"""
    
    # Set MLflow tracking URI (local file backend)
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("Starting MLOps experiments...")
    
    # Experiment 1: Baseline SVM on v1 dataset
    print("\n=== Experiment 1: Baseline SVM on v1 dataset ===")
    run_experiment(
        data_path="data/train_and_test2.csv",
        dataset_version="v1",
        model_type="svm",
        experiment_name="Baseline_Model"
    )
    
    # Experiment 2: Improved SVM on v2 dataset
    print("\n=== Experiment 2: Improved SVM on v2 dataset ===")
    run_experiment(
        data_path="data/train_and_test2_v2.csv",
        dataset_version="v2",
        model_type="svm",
        experiment_name="Improved_Model"
    )
    
    # Experiment 3: Hyperparameter tuned SVM on v2 dataset
    print("\n=== Experiment 3: Hyperparameter tuned SVM on v2 dataset ===")
    
    # First, get the data for hyperparameter tuning
    df = pd.read_csv("data/train_and_test2_v2.csv", encoding='utf-8')
    X, y = preprocess_data_v2(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Perform hyperparameter tuning
    best_params, best_score = hyperparameter_tuning(X_train, y_train, model_type='svm')
    print(f"Best hyperparameters: {best_params}")
    print(f"Best CV score: {best_score}")
    
    # Run experiment with best hyperparameters
    run_experiment(
        data_path="data/train_and_test2_v2.csv",
        dataset_version="v2",
        model_type="svm",
        experiment_name="Tuned_Model",
        hyperparams=best_params
    )
    
    # Experiment 4: Random Forest on v2 dataset
    print("\n=== Experiment 4: Random Forest on v2 dataset ===")
    run_experiment(
        data_path="data/train_and_test2_v2.csv",
        dataset_version="v2",
        model_type="rf",
        experiment_name="Random_Forest_Model"
    )
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()
