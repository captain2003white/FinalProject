# Data Versioning Documentation

## Overview
This document describes the data versioning strategy implemented using DVC (Data Version Control) for tracking different versions of the Titanic survival prediction dataset.

## Data Source
The original dataset is based on the famous Titanic passenger survival data, containing information about passengers including age, fare, sex, passenger class, and survival status.

## Dataset Versions

### Version 1 (v1) - Baseline Dataset
- **File**: `data/train_and_test2.csv`
- **DVC Hash**: `74cc1e4c4e7e30ec8a49abf227f5099c`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Description**: Original dataset with basic preprocessing
- **Size**: 1,309 samples, 28 features
- **Preprocessing**:
  - Removed zero columns (columns with all zeros)
  - Forward fill for missing Embarked values
  - Basic feature selection (removed Passengerid)

### Version 2 (v2) - Improved Dataset
- **File**: `data/train_and_test2_v2.csv`
- **DVC Hash**: `9e27aa4cc22e89453df8bd8f42b02133`
- **Git Commit**: `4147e16e93f0d90259461d7d66a63a01422a6ddd`
- **Description**: Enhanced dataset with advanced preprocessing and feature engineering
- **Size**: 576 samples, 31 features
- **Improvements from v1 → v2**:
  1. **Outlier Removal**: Removed fare outliers using IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR)
  2. **Better Missing Value Handling**: Filled missing ages with median age grouped by passenger class
  3. **Feature Engineering**:
     - Created AgeGroup categorical feature (Child, Teen, Adult, Middle, Senior)
     - Added FamilySize feature (sibsp + parch + 1)
     - Created FarePerPerson feature (fare / family_size)
  4. **Class Balancing**: Undersampled majority class to create balanced dataset
  5. **Synthetic Data Augmentation**: Added 50 synthetic samples with small random variations
  6. **Enhanced Preprocessing**: Better handling of categorical variables with label encoding

## Data Version Control Setup

### DVC Configuration
- **Local Storage**: `.dvc/cache/`
- **Remote Storage**: 
  - Primary: Dagshub (`https://dagshub.com/whitecaptain2003/FinalProject.dvc`)
  - Secondary: Azure (`azure://mlops-data-container`)

### Commands Used
```bash
# Initialize DVC tracking for v1
dvc add data/train_and_test2.csv

# Create and track v2 dataset
python create_v2_dataset.py
dvc add data/train_and_test2_v2.csv

# Push to remote storage
dvc push
```

## Data Lineage
```
Original Dataset (train_and_test2.csv)
    ↓
v1: Basic preprocessing
    ↓
v2: Advanced preprocessing + Feature Engineering
    ↓
Model Training Experiments
```

## Reproducibility
Each dataset version is tied to a specific git commit hash, ensuring that:
- Code changes can be traced back to specific data versions
- Experiments can be reproduced with exact data versions
- Data changes are tracked and versioned like code

## Usage in Experiments
- **Baseline Model**: Uses v1 dataset for establishing baseline performance
- **Improved Models**: Use v2 dataset for enhanced feature engineering and preprocessing
- **Model Comparison**: Both versions are used to demonstrate the impact of data quality improvements

## Data Privacy and Security
- Raw data files are not committed to git repository
- Only `.dvc` pointer files are tracked in git
- Actual data is stored in remote DVC storage
- `.gitignore` properly configured to exclude data files
