#!/usr/bin/env python3
"""
train_random_forest.py - Train and save Random Forest model for wind turbine anomaly detection.

Usage:
    python train_random_forest.py --input labeled_sensor_data.csv --output rf_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os
import argparse

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
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"\nDropped {rows_dropped} rows with missing values")
    
    # Define feature groups
    acf_features = [
        # Accelerometer ACF features
        'accel_x_acf_lag1', 'accel_x_acf_lag2', 'accel_x_acf_lag3', 'accel_x_acf_lag4',
        'accel_y_acf_lag1', 'accel_y_acf_lag2', 'accel_y_acf_lag3', 'accel_y_acf_lag4',
        'accel_z_acf_lag1', 'accel_z_acf_lag2', 'accel_z_acf_lag3', 'accel_z_acf_lag4',
        # Gyroscope ACF features
        'gyro_x_acf_lag1', 'gyro_x_acf_lag2', 'gyro_x_acf_lag3', 'gyro_x_acf_lag4',
        'gyro_y_acf_lag1', 'gyro_y_acf_lag2', 'gyro_y_acf_lag3', 'gyro_y_acf_lag4',
        'gyro_z_acf_lag1', 'gyro_z_acf_lag2', 'gyro_z_acf_lag3', 'gyro_z_acf_lag4'
    ]
    
    statistical_features = [
        # Accelerometer statistical features
        'accel_x_mean', 'accel_x_std', 'accel_x_min', 'accel_x_max', 'accel_x_median', 'accel_x_range',
        'accel_y_mean', 'accel_y_std', 'accel_y_min', 'accel_y_max', 'accel_y_median', 'accel_y_range',
        'accel_z_mean', 'accel_z_std', 'accel_z_min', 'accel_z_max', 'accel_z_median', 'accel_z_range',
        # Gyroscope statistical features
        'gyro_x_mean', 'gyro_x_std', 'gyro_x_min', 'gyro_x_max', 'gyro_x_median', 'gyro_x_range',
        'gyro_y_mean', 'gyro_y_std', 'gyro_y_min', 'gyro_y_max', 'gyro_y_median', 'gyro_y_range',
        'gyro_z_mean', 'gyro_z_std', 'gyro_z_min', 'gyro_z_max', 'gyro_z_median', 'gyro_z_range'
    ]
    
    # Combine all features
    feature_cols = acf_features + statistical_features
    
    # Verify all features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected features: {missing_cols}")
    
    # Separate features and target
    X = df[feature_cols]
    y = df['label']
    
    # Print feature summary
    print("\nFeature Summary:")
    print("=" * 50)
    print(f"Autocorrelation features: {len(acf_features)} (4 lags × 6 sensors)")
    print(f"Statistical features: {len(statistical_features)} (6 stats × 6 sensors)")
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature matrix shape: {X.shape}")
    
    print("\nClass distribution:")
    print(y.value_counts().sort_index())
    
    return X, y, feature_cols

def train_and_save_model(X, y, feature_cols, output_file):
    """Train and save the model."""
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
    
    print("\nModel Results:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Normal', 'Tempered Blade', 'Gearbox']))
    
    print("\nFeature Importance:")
    print("=" * 50)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    # Save model and features
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'feature_importance': importances.to_dict()
    }, output_file)
    
    print(f"\nSaved model and features to {output_file}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier for wind turbine anomaly detection"
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to labeled sensor data CSV file'
    )
    
    parser.add_argument(
        '--output',
        default='rf_model.pkl',
        help='Output model file (default: rf_model.pkl)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        X, y, feature_cols = load_and_prepare_data(args.input)
        
        print(f"\nTraining Random Forest with {len(feature_cols)} features...")
        accuracy = train_and_save_model(X, y, feature_cols, args.output)
        
        print(f"\nTraining completed successfully! Final accuracy: {accuracy:.4f}")
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 