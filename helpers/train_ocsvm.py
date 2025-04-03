#!/usr/bin/env python3
"""
Usage:
    python train_ocsvm.py input_file.csv
"""
import joblib
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def load_data(filepath):
    print("Loading data from " + filepath + "...")
    df = pd.read_csv(filepath)
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print("\nDropped " + str(rows_dropped) + " rows with missing values")
    
    return df
        
def select_features(df):
    # Define feature names for each sensor
    sensor_features = []
    for sensor in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        # Statistical features
        sensor_features.extend([
            sensor + '_mean',
            sensor + '_std',
            sensor + '_max',
            sensor + '_min',
            sensor + '_median',
            sensor + '_q1',
            sensor + '_q3',
            sensor + '_iqr',
            sensor + '_sum_abs',
            sensor + '_sum_squares'
        ])
        # ACF features
        for i in range(4):
            sensor_features.append(sensor + '_acf_lag' + str(i+1))
    
    X = df[sensor_features]
    return X

def train_model(X, nu=0.05):
    # Standardize the features to have zero mean and unit variance
    # This is important for SVM models which are sensitive to the scale of features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a One-Class SVM model for anomaly detection
    # - nu: Controls the proportion of outliers expected in the training data
    #   (higher values = more outliers allowed = more sensitive to anomalies)
    print("\nTraining One-Class SVM (nu=" + str(nu) + ")...")
    model = OneClassSVM(
        kernel='rbf',  # Radial Basis Function kernel for non-linear decision boundaries
        gamma='auto',  # Kernel coefficient - 'auto' uses 1/n_features
        nu=nu          # Expected proportion of outliers in the training data
    )
    model.fit(X_scaled)  # Train the model on the standardized features
    
    print("Training completed on " + str(len(X)) + " samples")    
    return model, scaler

def save_model(model, scaler, filepath='../models/model_svm.pkl'):
    model_dict = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_dict, filepath)
    print("\nModel and scaler saved to " + filepath)
   
def main():        
    df = load_data(sys.argv[1])
    X = select_features(df)
    model, scaler = train_model(X, nu=0.05)
    save_model(model, scaler)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    sys.exit(main()) 