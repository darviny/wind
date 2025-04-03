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
    print(f"Buffer data length: {len(data) if data is not None else 0}")
    
    if data is None or len(data) == 0:
        print("No data in buffer")
        return None
        
    data = np.array(data)
    print(f"Data shape: {data.shape}")
    
    # Extract features
    features = []
    
    # For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    for i in range(6):  # 6 sensors
        sensor_data = data[:, i]
        print(f"Sensor {i} data length: {len(sensor_data)}")
        
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
        acf_features = compute_acf_features(sensor_data)
        print(f"Sensor {i} ACF features: {acf_features}")
        features.extend(acf_features)
    
    features = np.array(features)
    print(f"Total features extracted: {len(features)}")
    return features

class OneClassSVMDetector:
    def __init__(self, model_path='models/model.pkl', scaler_path='models/scaler.pkl', sensitivity=0.5):
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                print("Loaded model from dictionary")
            else:
                self.model = model_data
                print("Loaded model directly")
            print(f"Model type: {type(self.model)}")
            
            # Set sensitivity (0.0 to 1.0, higher = less sensitive)
            self.sensitivity = max(0.0, min(1.0, sensitivity))
            print(f"SVM sensitivity set to {self.sensitivity}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        try:
            self.scaler = joblib.load(scaler_path)
            print("Successfully loaded scaler")
        except (FileNotFoundError, IOError):
            print(f"Scaler file {scaler_path} not found. Using identity scaling.")
            self.scaler = None
                
    def predict(self, features):
        if features is None or self.model is None:
            return 0.0
            
        features = np.array(features).reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        try:
            # Try to use decision_function if available
            if hasattr(self.model, 'decision_function'):
                score = self.model.decision_function(features)[0]
                print(f"Raw SVM score: {score}")
                
                # Normalize the score to be between -1 and 1
                # The decision_function returns signed distance to the separating hyperplane
                # Negative values indicate anomalies, positive values indicate normal samples
                normalized_score = score / np.abs(score) if score != 0 else 0
                print(f"Normalized SVM score: {normalized_score}")
                
                # Adjust threshold based on sensitivity
                # Higher sensitivity means we need a more negative score to detect anomaly
                threshold = -self.sensitivity
                print(f"Threshold: {threshold}")
                
                # Return the normalized score shifted by the threshold
                return float(normalized_score - threshold)
            else:
                # If no decision_function, use predict and return -1 for anomalies
                pred = self.model.predict(features)[0]
                score = -1.0 if pred == -1 else 1.0
                return float(score)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0

class RandomForestDetector:
    def __init__(self, model_path):
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_cols = model_data.get('features', [])
                self.scaler = model_data.get('scaler')
                print("Loaded Random Forest model from dictionary")
            else:
                self.model = model_data
                self.feature_cols = []
                self.scaler = None
                print("Loaded Random Forest model directly")
            print(f"Model type: {type(self.model)}")
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            self.model = None
            self.feature_cols = []
            self.scaler = None
        
    def predict(self, features):
        if features is None or self.model is None:
            return 0
            
        # Ensure features are in the correct order
        feature_vector = np.array(features)
        
        # Scale features using the saved scaler
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        
        try:
            # Make prediction and convert to integer
            prediction = self.model.predict(feature_vector)
            return int(prediction[0])
        except Exception as e:
            print(f"Error in Random Forest prediction: {e}")
            return 0