#!/usr/bin/env python3
"""
anomaly_detector.py - Module for detecting anomalies in MPU6050 sensor data.

This module provides functionality to check if sensor data features exceed
predefined thresholds for accelerometer, gyroscope, and temperature readings.
"""

# Default thresholds for sensor features
DEFAULT_THRESHOLDS = {
    # Accelerometer thresholds (m/s²)
    "accel_x_mean": 9.9,    # Allow for gravity on any axis
    "accel_y_mean": 9.9,    # Allow for gravity on any axis
    "accel_z_mean": 9.9,    # Allow for gravity on any axis
    "accel_x_std": 0.5,     # Increased to detect significant shaking
    "accel_y_std": 0.5,     # Increased to detect significant shaking
    "accel_z_std": 0.5,     # Increased to detect significant shaking
    "accel_x_max": 11.0,    # Allow for gravity plus movement
    "accel_y_max": 11.0,    # Allow for gravity plus movement
    "accel_z_max": 11.0,    # Allow for gravity plus movement
    
    # Gyroscope thresholds (rad/s)
    "gyro_x_mean": 0.1,
    "gyro_y_mean": 0.1,
    "gyro_z_mean": 0.1,
    "gyro_x_std": 0.05,
    "gyro_y_std": 0.05,
    "gyro_z_std": 0.05,
    "gyro_x_max": 0.5,
    "gyro_y_max": 0.5,
    "gyro_z_max": 0.5,
    
    # Temperature thresholds (°C)
    "temp_mean": 50,    # Maximum average temperature
    "temp_std": 2,      # Maximum temperature variation
    "temp_max": 85      # Maximum absolute temperature
}


def check_anomaly(features, thresholds=None):
    """
    Check if any feature exceeds its corresponding threshold.
    
    Args:
        features (dict): Dictionary of feature names and their values.
            Example: {"accel_x_mean": 0.01, "gyro_y_std": 0.005, "temp_mean": 25}
        thresholds (dict, optional): Dictionary of feature names and their threshold values.
            If None, DEFAULT_THRESHOLDS will be used.
    
    Returns:
        tuple: (is_anomaly, exceeded_features)
            - is_anomaly (bool): True if any feature exceeds its threshold
            - exceeded_features (dict): Features that exceeded their thresholds
    """
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # Track features that exceed thresholds
    exceeded_features = {}
    
    # Check each feature against its threshold
    for feature, value in features.items():
        # Skip if this feature doesn't have a threshold
        if feature not in thresholds:
            continue
        
        # Check if the feature exceeds its threshold
        threshold = thresholds[feature]
        if value > threshold:
            exceeded_features[feature] = (value, threshold)
    
    # Return True if any features exceeded thresholds
    is_anomaly = len(exceeded_features) > 0
    
    return is_anomaly, exceeded_features


def analyze_readings(readings):
    """
    Calculate statistical features from a list of sensor readings.
    
    Args:
        readings (list): List of dictionaries containing sensor readings.
            Example: [
                {
                    'acceleration': {'x': 0.1, 'y': 0.2, 'z': 9.8},
                    'gyro': {'x': 0.01, 'y': 0.02, 'z': 0.01},
                    'temperature': 25.0
                },
                ...
            ]
    
    Returns:
        dict: Dictionary of calculated features.
    """
    import statistics
    
    # Initialize empty lists for each measurement
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    temperatures = []
    
    # Extract values from readings
    for reading in readings:
        # Acceleration values
        accel = reading['acceleration']
        accel_x.append(accel['x'])
        accel_y.append(accel['y'])
        accel_z.append(accel['z'])
        
        # Gyroscope values
        gyro = reading['gyro']
        gyro_x.append(gyro['x'])
        gyro_y.append(gyro['y'])
        gyro_z.append(gyro['z'])
        
        # Temperature values
        temperatures.append(reading['temperature'])
    
    # Helper function for standard deviation
    def safe_stdev(values):
        return statistics.stdev(values) if len(values) > 1 else 0
    
    # Calculate features
    features = {
        # Accelerometer features
        "accel_x_mean": statistics.mean(accel_x),
        "accel_y_mean": statistics.mean(accel_y),
        "accel_z_mean": statistics.mean(accel_z),
        "accel_x_std": safe_stdev(accel_x),
        "accel_y_std": safe_stdev(accel_y),
        "accel_z_std": safe_stdev(accel_z),
        "accel_x_max": max(accel_x),
        "accel_y_max": max(accel_y),
        "accel_z_max": max(accel_z),
        
        # Gyroscope features
        "gyro_x_mean": statistics.mean(gyro_x),
        "gyro_y_mean": statistics.mean(gyro_y),
        "gyro_z_mean": statistics.mean(gyro_z),
        "gyro_x_std": safe_stdev(gyro_x),
        "gyro_y_std": safe_stdev(gyro_y),
        "gyro_z_std": safe_stdev(gyro_z),
        "gyro_x_max": max(gyro_x),
        "gyro_y_max": max(gyro_y),
        "gyro_z_max": max(gyro_z),
        
        # Temperature features
        "temp_mean": statistics.mean(temperatures),
        "temp_std": safe_stdev(temperatures),
        "temp_max": max(temperatures)
    }
    
    return features


# Example usage
if __name__ == "__main__":
    # Example sensor readings
    sample_readings = [
        {
            'acceleration': {'x': 0.01, 'y': 0.02, 'z': 9.81},
            'gyro': {'x': 0.01, 'y': 0.02, 'z': 0.01},
            'temperature': 25.0
        },
        {
            'acceleration': {'x': 0.02, 'y': 0.03, 'z': 9.82},
            'gyro': {'x': 0.02, 'y': 0.01, 'z': 0.02},
            'temperature': 25.1
        }
    ]
    
    # Calculate features from the readings
    features = analyze_readings(sample_readings)
    print("Calculated features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.4f}")
    
    # Check for anomalies using default thresholds
    is_anomaly, exceeded = check_anomaly(features)
    
    print(f"\nAnomaly detected: {is_anomaly}")
    if is_anomaly:
        print("Exceeded thresholds:")
        for feature, (value, threshold) in exceeded.items():
            print(f"  {feature}: {value:.4f} exceeds threshold {threshold}")