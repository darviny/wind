#!/usr/bin/env python3
"""
Usage:
    python train_random_forest.py input_file.csv output_file.pkl
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import sys

def load_and_prepare_data(filepath):  
    df = pd.read_csv(filepath)
    print("Initial dataset shape: " + str(df.shape))
    
    # Handle missing values by dropping rows with any NaN values
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print("\nDropped " + str(rows_dropped) + " rows with missing values")
    
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
    
    # Extract features (X) and target labels (y) from the dataset
    X = df[sensor_features]
    y = df['label']
    
    return X, y, sensor_features

def train_and_save_model(X, y, feature_cols, output_file):
    # Split the data into training and testing sets
    # - test_size=0.2: Use 20% of data for testing
    # - random_state=42: For reproducibility
    # - stratify=y: Ensure balanced class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train a Random Forest classifier
    # - n_estimators=100: Number of decision trees in the forest
    # - random_state=42: For reproducibility
    # - n_jobs=-1: Use all available CPU cores for parallel processing
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print model performance metrics
    print("Model Results:")
    print("================================================")
    print("Accuracy: " + str(accuracy))
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Tempered Blade', 'Gearbox']))
    
    print("Feature Importance:")
    print("================================================")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    # Save model, features, scaler, and feature importance
    model_dict = {
        'model': model,
        'feature_names': feature_cols,
        'scaler': scaler,
        'scaler_feature_names': feature_cols,
        'feature_importance': importances.to_dict()
    }
    joblib.dump(model_dict, output_file)
    
    print("\nSaved model and features to " + output_file)
    return accuracy

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    X, y, feature_cols = load_and_prepare_data(input_file)
    print("\nTraining Random Forest with " + str(len(feature_cols)) + " features...")
    accuracy = train_and_save_model(X, y, feature_cols, output_file)
    print("\nTraining completed successfully! Final accuracy: " + str(accuracy))
    return 0

if __name__ == "__main__":
    sys.exit(main()) 