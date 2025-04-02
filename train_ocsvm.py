#!/usr/bin/env python3
"""
train_ocsvm.py - Train a One-Class SVM model for anomaly detection on sensor data.

Usage:
    python train_ocsvm.py input_file.csv
Example:
    python train_ocsvm.py sensor_data_normal_aggregated.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import sys
import os

def load_data(filepath):
    """Load and prepare the aggregated sensor data."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found!")
            return None
            
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\nWarning: Found missing values:")
            print(missing[missing > 0])
            print("Dropping rows with missing values...")
            df = df.dropna()
        
        print(f"\nLoaded data shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def select_features(df):
    """Select relevant features for training."""
    # Define feature columns for both ACF and statistical features
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
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None
    
    X = df[feature_cols]
    
    # Print feature summary
    print("\nFeature Selection Summary:")
    print("=" * 50)
    print(f"Autocorrelation features: {len(acf_features)}")
    print(f"Statistical features: {len(statistical_features)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nFeature matrix shape: {X.shape}")
    
    # Print sample of each feature type
    print("\nSample ACF features:", acf_features[:3], "...")
    print("Sample statistical features:", statistical_features[:3], "...")
    
    return X

def train_model(X, nu=0.05):
    """
    Train One-Class SVM model and scaler.
    
    Args:
        X: Feature matrix
        nu: Proportion of training errors (outliers) allowed
    """
    try:
        # Initialize and fit scaler
        print("\nStandardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize and train model
        print(f"\nTraining One-Class SVM (nu={nu})...")
        model = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=nu
        )
        model.fit(X_scaled)
        
        print(f"Training completed on {len(X)} samples")
        
        # Basic evaluation on training data
        predictions = model.predict(X_scaled)
        n_inliers = sum(predictions == 1)
        n_outliers = sum(predictions == -1)
        
        print("\nTraining set predictions:")
        print(f"Inliers: {n_inliers} ({n_inliers/len(X)*100:.1f}%)")
        print(f"Outliers: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

def save_model(model, scaler, filepath='model.pkl'):
    """Save the trained model and scaler."""
    try:
        model_dict = {
            'model': model,
            'scaler': scaler
        }
        
        joblib.dump(model_dict, filepath)
        print(f"\nModel and scaler saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python train_ocsvm.py input_file.csv")
        return 1
        
    input_file = sys.argv[1]
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return 1
    
    # Select features
    X = select_features(df)
    if X is None:
        return 1
    
    # Train model
    model, scaler = train_model(X, nu=0.05)
    if model is None or scaler is None:
        return 1
    
    # Save model
    if not save_model(model, scaler):
        return 1
    
    print("\nTraining completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 