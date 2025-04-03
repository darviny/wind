#!/usr/bin/env python3
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_acf_features(signal, n_lags=4):
    # Remove mean
    signal = signal - np.mean(signal)
    
    # Compute autocorrelation
    acf = np.correlate(signal, signal, mode='full')
    
    # Keep only positive lags
    acf = acf[len(signal)-1:]
    
    # Normalize by zero lag
    if acf[0] > 0:
        acf = acf / acf[0]
    
    # Return first n_lags values
    return acf[1:n_lags+1].tolist()

def extract_features(buffer):
    # Get the latest window of data
    data = buffer.get_latest_window()
    if data is None or len(data) == 0:
        return None
        
    data = np.array(data)
    
    # Extract features
    features = []
    
    # For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    for i in range(6):  # 6 sensors
        sensor_data = data[:, i]
        
        # Add statistical features
        features.extend([
            np.mean(sensor_data),
            np.std(sensor_data),
            np.max(sensor_data),
            np.min(sensor_data),
            np.median(sensor_data),
            np.percentile(sensor_data, 25),  # q1
            np.percentile(sensor_data, 75),  # q3
            np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25),  # iqr
            np.sum(np.abs(sensor_data)),  # sum_abs
            np.sum(np.square(sensor_data))  # sum_squares
        ])
        
        # Add ACF features
        features.extend(compute_acf_features(sensor_data))
    
    return np.array(features)

class OneClassSVMDetector:
    def __init__(self, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Successfully loaded model and scaler")
                
    def predict(self, features):
        if features is None:
            return 0.0
            
        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        
        # Get decision function score (negative distance to hyperplane)
        # More negative score = more likely to be an anomaly
        return float(self.model.decision_function(features)[0])

class RandomForestDetector:
    def __init__(self, model_path):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['features']
        self.scaler = model_data.get('scaler')  # Get scaler from saved model data
        
    def predict(self, features):
        # Ensure features are in the correct order
        feature_vector = np.array(features)
        
        # Scale features using the saved scaler
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Make prediction
        return self.model.predict(feature_vector)[0]