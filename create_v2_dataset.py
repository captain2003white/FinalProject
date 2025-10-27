import pandas as pd
import numpy as np
import os

def create_v2_dataset():
    """Create an improved v2 dataset with data cleaning and preprocessing"""
    
    # Read the original dataset
    df = pd.read_csv('data/train_and_test2.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Create v2 improvements:
    # 1. Remove outliers based on Fare (using IQR method)
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    df_cleaned = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
    
    print(f"After removing Fare outliers: {df_cleaned.shape}")
    
    # 2. Handle missing values in Age more intelligently
    # Fill missing ages with median age by passenger class
    df_cleaned['Age'] = df_cleaned.groupby('Pclass')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 3. Create additional features for better model performance
    # Age groups
    df_cleaned['AgeGroup'] = pd.cut(df_cleaned['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare per person (considering family size)
    df_cleaned['FamilySize'] = df_cleaned['sibsp'] + df_cleaned['Parch'] + 1
    df_cleaned['FarePerPerson'] = df_cleaned['Fare'] / df_cleaned['FamilySize']
    
    # 4. Balance the dataset by undersampling the majority class
    survived_counts = df_cleaned['2urvived'].value_counts()
    min_count = min(survived_counts)
    
    # Sample equal number from each class
    df_balanced = pd.concat([
        df_cleaned[df_cleaned['2urvived'] == 0].sample(n=min_count, random_state=42),
        df_cleaned[df_cleaned['2urvived'] == 1].sample(n=min_count, random_state=42)
    ]).reset_index(drop=True)
    
    print(f"After balancing classes: {df_balanced.shape}")
    print(f"Class distribution: {df_balanced['2urvived'].value_counts()}")
    
    # 5. Add synthetic samples using SMOTE-like approach (simplified)
    # For demonstration, we'll duplicate some samples with small variations
    synthetic_samples = []
    for _ in range(50):  # Add 50 synthetic samples
        sample = df_balanced.sample(1).iloc[0].copy()
        # Add small random noise to continuous features
        sample['Age'] += np.random.normal(0, 1)
        sample['Fare'] += np.random.normal(0, 5)
        sample['FarePerPerson'] = sample['Fare'] / sample['FamilySize']
        synthetic_samples.append(sample)
    
    df_synthetic = pd.concat([df_balanced, pd.DataFrame(synthetic_samples)], ignore_index=True)
    
    print(f"After adding synthetic samples: {df_synthetic.shape}")
    
    # Save the v2 dataset
    os.makedirs('data', exist_ok=True)
    df_synthetic.to_csv('data/train_and_test2_v2.csv', index=False)
    
    print("V2 dataset created successfully!")
    print(f"Final dataset shape: {df_synthetic.shape}")
    print(f"Class distribution: {df_synthetic['2urvived'].value_counts()}")
    
    return df_synthetic

if __name__ == "__main__":
    create_v2_dataset()
