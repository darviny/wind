#!/usr/bin/env python3
"""
Usage:
    python train_ocsvm.py input_file.csv
"""
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def load_data(file_path):
    """Load sensor data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def extract_features(data):
    """Extract features from the pre-computed features file."""
    # The features file already contains the mean and std for each sensor
    # We just need to get the values in the correct order
    features = []
    feature_names = []
    
    # Sensor names and their corresponding feature suffixes
    sensors = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    suffixes = ['_mean', '_std']
    
    # Extract features in the correct order
    for sensor in sensors:
        for suffix in suffixes:
            col_name = sensor + suffix
            if col_name in data.columns:
                features.append(data[col_name].values[0])  # Get the first row's value
                feature_names.append(col_name)
            else:
                print(f"Warning: Column {col_name} not found in data")
    
    features = np.array(features)
    print(f"Total features extracted: {len(features)}")
    print(f"Feature names: {feature_names}")
    return features, feature_names

def train_and_save_model(features, feature_names, model_path, scaler_path):
    """Train One-Class SVM model and save it with feature names."""
    try:
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.reshape(1, -1))
        
        # Train model
        model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        model.fit(scaled_features)
        
        # Save model with feature names
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler with feature names
        scaler_data = {
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(scaler_data, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    except Exception as e:
        print(f"Error in training/saving: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_ocsvm.py <input_csv> <output_model>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_model = sys.argv[2]
    output_scaler = output_model.replace('model_svm.pkl', 'scaler.pkl')
    
    print(f"Input file: {input_file}")
    print(f"Output model: {output_model}")
    print(f"Output scaler: {output_scaler}")
    
    # Load and process data
    data = load_data(input_file)
    features, feature_names = extract_features(data)
    
    # Train and save model
    train_and_save_model(features, feature_names, output_model, output_scaler)
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 