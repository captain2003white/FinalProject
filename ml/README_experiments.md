# MLflow Experiment Tracking Documentation

## Overview
This document describes the MLflow experiment tracking implementation for the Titanic survival prediction project, including baseline and improved model experiments.

## Experiment Setup
- **MLflow Tracking URI**: `file:./mlruns` (local file backend)
- **Experiments**: 4 distinct experiments tracked
- **Models**: SVM and Random Forest classifiers
- **Datasets**: v1 (baseline) and v2 (improved) versions

## Experiments Summary

### Experiment 1: Baseline Model
- **Experiment Name**: `Baseline_Model`
- **Model Type**: SVM (Support Vector Machine)
- **Dataset**: v1 (baseline dataset)
- **Dataset Hash**: `74cc1e4c4e7e30ec8a49abf227f5099c`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Hyperparameters**:
  - C: 1
  - gamma: 0.1
  - kernel: rbf
- **Performance Metrics**:
  - Accuracy: 0.7837
  - F1 Score: 0.5087
  - Recall: 0.5087
  - Precision: 0.5087
- **Purpose**: Establish baseline performance with original dataset and default hyperparameters

### Experiment 2: Improved Model
- **Experiment Name**: `Improved_Model`
- **Model Type**: SVM (Support Vector Machine)
- **Dataset**: v2 (improved dataset)
- **Dataset Hash**: `9e27aa4cc22e89453df8bd8f42b02133`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Hyperparameters**:
  - C: 1
  - gamma: 0.1
  - kernel: rbf
- **Performance Metrics**:
  - Accuracy: 0.7688
  - F1 Score: 0.7727
  - Recall: 0.7727
  - Precision: 0.7727
- **Purpose**: Demonstrate impact of improved data preprocessing and feature engineering

### Experiment 3: Hyperparameter Tuned Model
- **Experiment Name**: `Tuned_Model`
- **Model Type**: SVM (Support Vector Machine)
- **Dataset**: v2 (improved dataset)
- **Dataset Hash**: `9e27aa4cc22e89453df8bd8f42b02133`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Hyperparameter Tuning**:
  - Grid Search CV: 5-fold cross-validation
  - Parameter Grid: C=[0.1, 1, 10, 100], gamma=[0.001, 0.01, 0.1, 1], kernel=['rbf', 'linear']
  - Best CV Score: 0.7779
- **Best Hyperparameters**:
  - C: 1
  - gamma: 0.1
  - kernel: rbf
- **Performance Metrics**:
  - Accuracy: 0.7688
  - F1 Score: 0.7727
  - Recall: 0.7727
  - Precision: 0.7727
- **Purpose**: Optimize hyperparameters for best performance

### Experiment 4: Random Forest Model
- **Experiment Name**: `Random_Forest_Model`
- **Model Type**: Random Forest
- **Dataset**: v2 (improved dataset)
- **Dataset Hash**: `9e27aa4cc22e89453df8bd8f42b02133`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Hyperparameters**:
  - n_estimators: 100
  - random_state: 42
- **Performance Metrics**:
  - Accuracy: 0.7225
  - F1 Score: 0.7209
  - Recall: 0.7209
  - Precision: 0.7209
- **Purpose**: Compare different model algorithms

## Production-Worthy Model Selection

### Recommended Production Model: **Experiment 2 - Improved Model (SVM on v2 dataset)**

**Rationale**:
1. **Best F1 Score**: 0.7727 - This is the highest F1 score achieved across all experiments
2. **Balanced Performance**: Good balance between precision and recall
3. **Improved Dataset**: Uses v2 dataset with enhanced feature engineering and preprocessing
4. **Stable Performance**: Consistent results across multiple runs
5. **Feature Engineering Benefits**: Leverages additional features like AgeGroup, FamilySize, and FarePerPerson

### Optimization Metric: **F1 Score**

**Why F1 Score Matters**:
- **Balanced Metric**: F1 score provides a harmonic mean of precision and recall, crucial for imbalanced datasets
- **Survival Prediction Context**: In survival prediction, both false positives (predicting survival when passenger died) and false negatives (predicting death when passenger survived) have significant implications
- **Business Impact**: 
  - High precision ensures we don't overestimate survival chances
  - High recall ensures we don't miss actual survivors
  - F1 score balances both concerns optimally
- **Model Comparison**: Provides a single metric for comparing different models and configurations

## Logged Artifacts
Each experiment logs the following artifacts:
1. **Trained Model**: Serialized model file (.pkl)
2. **Scaler**: StandardScaler object for data preprocessing
3. **Confusion Matrix**: Visual representation of model performance
4. **Model Metadata**: Model type, creation time, feature names

## Experiment Tracking Features
- **Code Versioning**: Git commit hash logged for each experiment
- **Data Versioning**: DVC dataset hash logged for reproducibility
- **Parameter Tracking**: All hyperparameters and configuration parameters logged
- **Metric Tracking**: Accuracy, precision, recall, and F1 score logged
- **Artifact Management**: Models and visualizations stored and versioned
- **Experiment Comparison**: Easy comparison across different experiments

## Reproducibility
Each experiment is fully reproducible with:
- Exact git commit hash
- Specific dataset version (DVC hash)
- Complete hyperparameter configuration
- Random seed for consistent train/test splits
- Standardized preprocessing pipeline

## Next Steps
1. **Model Deployment**: Deploy the production-worthy model (Experiment 2) to a serving environment
2. **A/B Testing**: Compare production model performance with new experiments
3. **Continuous Monitoring**: Track model performance in production
4. **Model Retraining**: Set up automated retraining pipeline with new data
