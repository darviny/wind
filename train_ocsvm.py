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
    # Define feature columns based on the actual columns in your data
    feature_cols = [
        'accel_x_mean', 'accel_x_std',
        'accel_y_mean', 'accel_y_std',
        'accel_z_mean', 'accel_z_std',
        'gyro_x_mean', 'gyro_x_std',
        'gyro_y_mean', 'gyro_y_std',
        'gyro_z_mean', 'gyro_z_std'
    ]
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None
    
    X = df[feature_cols]
    print(f"\nFeature matrix shape: {X.shape}")
    print("Selected features:", feature_cols)
    
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