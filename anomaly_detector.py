#!/usr/bin/env python3
"""
anomaly_detector.py - Module for detecting anomalies in sensor data.

This module provides functionality to check if sensor data features exceed
predefined thresholds, which may indicate anomalous behavior.
"""

# Default thresholds for common acceleration features
DEFAULT_THRESHOLDS = {
    "x_mean": 0.5,    # Mean acceleration in X-axis (m/s²)
    "y_mean": 0.5,    # Mean acceleration in Y-axis (m/s²)
    "z_mean": 9.9,    # Mean acceleration in Z-axis (m/s²) - ~9.8 is gravity
    "x_std": 0.1,     # Standard deviation of X acceleration (m/s²)
    "y_std": 0.1,     # Standard deviation of Y acceleration (m/s²)
    "z_std": 0.1,     # Standard deviation of Z acceleration (m/s²)
    "x_max": 1.0,     # Maximum X acceleration (m/s²)
    "y_max": 1.0,     # Maximum Y acceleration (m/s²)
    "z_max": 10.5,    # Maximum Z acceleration (m/s²)
}


def check_anomaly(features, thresholds=None):
    """
    Check if any feature exceeds its corresponding threshold.
    
    Args:
        features (dict): Dictionary of feature names and their values.
            Example: {"x_mean": 0.01, "y_std": 0.005, "z_max": 10.2}
        thresholds (dict, optional): Dictionary of feature names and their threshold values.
            If None, DEFAULT_THRESHOLDS will be used. If a feature in 'features'
            doesn't have a corresponding threshold, it will be ignored.
    
    Returns:
        tuple: (is_anomaly, exceeded_features)
            - is_anomaly (bool): True if any feature exceeds its threshold, False otherwise.
            - exceeded_features (dict): Dictionary of features that exceeded their thresholds,
              with the feature name as key and a tuple of (value, threshold) as value.
    
    Example:
        >>> features = {"x_std": 0.2, "y_std": 0.05, "z_std": 0.08}
        >>> is_anomaly, exceeded = check_anomaly(features)
        >>> print(f"Anomaly detected: {is_anomaly}")
        >>> if is_anomaly:
        ...     for feature, (value, threshold) in exceeded.items():
        ...         print(f"{feature}: {value} exceeds threshold {threshold}")
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
        readings (list): List of dictionaries, each containing 'x', 'y', 'z' values.
            Example: [{'x': 0.1, 'y': 0.2, 'z': 9.8}, {'x': 0.15, 'y': 0.22, 'z': 9.82}]
    
    Returns:
        dict: Dictionary of calculated features.
    """
    import statistics
    
    # Initialize empty lists for each axis
    x_values = []
    y_values = []
    z_values = []
    
    # Extract values from readings
    for reading in readings:
        x_values.append(reading['x'])
        y_values.append(reading['y'])
        z_values.append(reading['z'])
    
    # Calculate features
    features = {
        "x_mean": statistics.mean(x_values),
        "y_mean": statistics.mean(y_values),
        "z_mean": statistics.mean(z_values),
        "x_std": statistics.stdev(x_values) if len(x_values) > 1 else 0,
        "y_std": statistics.stdev(y_values) if len(y_values) > 1 else 0,
        "z_std": statistics.stdev(z_values) if len(z_values) > 1 else 0,
        "x_max": max(x_values),
        "y_max": max(y_values),
        "z_max": max(z_values)
    }
    
    return features


# Example usage
if __name__ == "__main__":
    # Example sensor readings (x, y, z accelerations)
    sample_readings = [
        {'x': 0.01, 'y': 0.02, 'z': 9.81},
        {'x': 0.02, 'y': 0.03, 'z': 9.82},
        {'x': 0.15, 'y': 0.05, 'z': 9.80},
        {'x': 0.25, 'y': 0.02, 'z': 9.83},
        {'x': 0.01, 'y': 0.01, 'z': 9.79}
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
    
    # Example with custom thresholds
    custom_thresholds = {
        "x_std": 0.05,
        "y_std": 0.05,
        "z_std": 0.05
    }
    print("\nChecking with custom thresholds:")
    is_anomaly, exceeded = check_anomaly(features, custom_thresholds)
    print(f"Anomaly detected: {is_anomaly}")
    if is_anomaly:
        for feature, (value, threshold) in exceeded.items():
            print(f"  {feature}: {value:.4f} exceeds threshold {threshold}")