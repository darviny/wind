#!/usr/bin/env python3
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(buffer):
    # Get the latest window of data
    data = buffer.get_latest_window()
    print(f"Buffer data length: {len(data) if data is not None else 0}")
    
    if data is None or len(data) == 0:
        print("No data in buffer")
        return None
        
    data = np.array(data)
    print(f"Data shape: {data.shape}")
    
    # Extract only basic statistical features
    features = []
    feature_names = []
    
    # For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    sensor_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for sensor_name in sensor_names:
        sensor_data = data[:, sensor_names.index(sensor_name)]
        print(f"Sensor {sensor_name} data length: {len(sensor_data)}")
        
        # Only use mean and standard deviation
        features.extend([
            np.mean(sensor_data),      # mean
            np.std(sensor_data),       # std
        ])
        feature_names.extend([
            f'{sensor_name}_mean',
            f'{sensor_name}_std',
        ])
    
    features = np.array(features)
    print(f"Total features extracted: {len(features)}")
    print(f"Feature names: {feature_names}")
    return features, feature_names

class OneClassSVMDetector:
    def __init__(self, model_path='models/model_svm.pkl', scaler_path='models/scaler.pkl', sensitivity=0.5):
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', [])
                print("Loaded model from dictionary")
                print(f"Feature names: {self.feature_names}")
            else:
                self.model = model_data
                self.feature_names = []
                print("Loaded model directly")
            print(f"Model type: {type(self.model)}")
            
            # Set sensitivity (0.0 to 1.0, higher = less sensitive)
            self.sensitivity = max(0.0, min(1.0, sensitivity))
            print(f"SVM sensitivity set to {self.sensitivity}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        try:
            scaler_data = joblib.load(scaler_path)
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data['scaler']
                self.scaler_feature_names = scaler_data.get('feature_names', [])
                print("Loaded scaler from dictionary")
                print(f"Scaler feature names: {self.scaler_feature_names}")
            else:
                self.scaler = scaler_data
                self.scaler_feature_names = []
                print("Loaded scaler directly")
            print("Successfully loaded scaler")
        except (FileNotFoundError, IOError):
            print(f"Scaler file {scaler_path} not found. Using identity scaling.")
            self.scaler = None
                
    def predict(self, features):
        if features is None or self.model is None:
            return 0.0
            
        features = np.array(features).reshape(1, -1)
        if self.scaler is not None:
            try:
                # Convert features to DataFrame with feature names if available
                if hasattr(self, 'scaler_feature_names') and self.scaler_feature_names:
                    import pandas as pd
                    print(f"Using scaler feature names: {self.scaler_feature_names}")
                    features_df = pd.DataFrame(features, columns=self.scaler_feature_names)
                    features = self.scaler.transform(features_df)
                else:
                    print("No feature names available, using direct transformation")
                    features = self.scaler.transform(features)
            except Exception as e:
                print(f"Error in scaling: {e}")
                return 0.0
        
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