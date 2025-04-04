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
                features.append(data[col_name].values)  # Get all values for this feature
                feature_names.append(col_name)
            else:
                print(f"Warning: Column {col_name} not found in data")
    
    # Transpose to get shape (n_samples, n_features)
    features = np.array(features).T
    print(f"Total features extracted: {features.shape[1]}")
    print(f"Number of samples: {features.shape[0]}")
    print(f"Feature names: {feature_names}")
    return features, feature_names

def train_and_save_model(features, feature_names, model_path, scaler_path):
    """Train One-Class SVM model and save it with feature names."""
    try:
        # Convert features to DataFrame with feature names
        features_df = pd.DataFrame(features, columns=feature_names)
        print("\nFeatures DataFrame shape:", features_df.shape)
        print("\nFirst 5 rows of features:")
        print(features_df.head())
        
        # Scale features using the DataFrame to preserve feature names
        scaler = StandardScaler()
        scaler.fit(features_df)  # Fit on DataFrame to learn feature names
        scaled_features = scaler.transform(features_df)
        
        # Train model with more robust parameters
        # nu: controls the fraction of outliers (lower = more strict)
        # gamma: controls the influence of each training example (lower = smoother decision boundary)
        model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
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