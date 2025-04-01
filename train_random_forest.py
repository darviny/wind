#!/usr/bin/env python3
"""
train_random_forest.py - Train and save both versions of the Random Forest model.

This script trains a model to classify wind turbine operation into three categories:
- Normal operation (0)
- Tempered blade anomaly (1)
- Gearbox imbalance anomaly (2)

The model is trained on engineered features from sensor data and saved for later use.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os

def load_and_prepare_data(filepath):
    """
    Load and prepare the dataset, handling missing values.
    
    Args:
        filepath (str): Path to the labeled sensor data CSV
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    print("\nStep 1: Loading dataset...")
    
    # Load data
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Initial dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"\nDropped {rows_dropped} rows with missing values")
    
    # Separate features and target
    target_col = 'label'
    feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp', 'condition']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("\nFeature columns:", feature_cols)
    print(f"Feature matrix shape: {X.shape}")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())
    
    return X, y, feature_cols

def train_and_save_model(X, y, feature_cols, model_prefix):
    """Train and save a model with given features."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_prefix} Results:")
    print("=" * 50)
    print(f"Features used: {feature_cols}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Normal', 'Tempered Blade', 'Gearbox']))
    
    # Save model and features
    model_file = f'{model_prefix}_model.pkl'
    features_file = f'{model_prefix}_features.json'
    
    joblib.dump(model, model_file)
    with open(features_file, 'w') as f:
        json.dump(feature_cols, f)
    
    print(f"\nSaved model as {model_file}")
    print(f"Saved features as {features_file}")
    
    return accuracy

def main():
    print("Training and saving both model versions...")
    
    # Load data
    df = pd.read_csv('labeled_sensor_data.csv')
    y = df['label']
    
    # Define feature sets
    all_features = [col for col in df.columns if col not in ['label', 'timestamp', 'condition']]
    vibration_features = [col for col in all_features if 'temperature' not in col]
    
    # Train and save model with temperature
    X_all = df[all_features]
    acc_all = train_and_save_model(X_all, y, all_features, "with_temp")
    
    # Train and save model without temperature
    X_vib = df[vibration_features]
    acc_vib = train_and_save_model(X_vib, y, vibration_features, "without_temp")
    
    # Print comparison summary
    print("\nComparison Summary:")
    print("=" * 50)
    print(f"Accuracy with temperature: {acc_all:.4f}")
    print(f"Accuracy without temperature: {acc_vib:.4f}")
    print(f"Difference: {(acc_vib - acc_all)*100:.2f}%")
    
    print("\nBoth models have been saved successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 