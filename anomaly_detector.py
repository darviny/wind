#!/usr/bin/env python3
"""
anomaly_detector.py - Anomaly detection module supporting both One-Class SVM
and threshold-based detection methods.

The module attempts to load a pre-trained One-Class SVM model from 'model.pkl'.
If the model file is not found, it falls back to threshold-based detection.
"""

import os
import logging
import numpy as np
from typing import Union, List, Tuple
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OneClassSVMDetector:
    """
    Anomaly detector using One-Class SVM with fallback to threshold-based detection.
    
    Attributes:
        model: Loaded One-Class SVM model or None if using fallback
        scaler: StandardScaler for feature normalization or None if not used
        using_fallback (bool): Indicates if using fallback threshold detection
    """
    
    def __init__(self, model_path: str = 'model.pkl', scaler_path: str = 'scaler.pkl'):
        """
        Initialize the detector by loading the model and scaler if available.
        
        Args:
            model_path: Path to the saved One-Class SVM model
            scaler_path: Path to the saved StandardScaler
        """
        self.model = None
        self.scaler = None
        self.using_fallback = False
        
        try:
            self.model = joblib.load(model_path)
            logger.info("Successfully loaded One-Class SVM model")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Successfully loaded scaler")
                
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load model/scaler: {str(e)}")
            logger.warning("Falling back to threshold-based detection")
            self.using_fallback = True
            
        # Fallback thresholds
        self.acc_threshold = 2.0  # Default acceleration threshold (in g)
        self.gyro_threshold = 100.0  # Default gyroscope threshold (in deg/s)
    
    def predict(self, features: Union[List[float], np.ndarray]) -> Tuple[bool, float, dict]:
        """
        Predict if the input features represent an anomaly.
        
        Args:
            features: Array-like of features [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            Tuple of (is_anomaly: bool, score: float, details: dict)
            - is_anomaly: True if anomaly detected
            - score: Anomaly score
            - details: Dictionary with threshold details
        """
        features = np.array(features).reshape(1, -1)
        
        if not self.using_fallback:
            # One-Class SVM model detection
            if self.scaler:
                features = self.scaler.transform(features)
            
            score = float(self.model.decision_function(features)[0])
            is_anomaly = self.model.predict(features)[0] == -1
            
            return is_anomaly, score, {"model_score": score}
        
        else:
            # Fallback threshold-based detection
            acc_values = features[0, :3]
            gyro_values = features[0, 3:6]
            
            # Calculate individual ratios
            acc_ratios = np.abs(acc_values) / self.acc_threshold
            gyro_ratios = np.abs(gyro_values) / self.gyro_threshold
            
            # Get maximum ratios
            max_acc_ratio = np.max(acc_ratios)
            max_gyro_ratio = np.max(gyro_ratios)
            
            # Overall score is the maximum ratio
            score = max(max_acc_ratio, max_gyro_ratio)
            is_anomaly = score > 1.0
            
            # Prepare detailed threshold information
            details = {
                "acceleration": {
                    "x": {"value": float(acc_values[0]), "ratio": float(acc_ratios[0])},
                    "y": {"value": float(acc_values[1]), "ratio": float(acc_ratios[1])},
                    "z": {"value": float(acc_values[2]), "ratio": float(acc_ratios[2])},
                    "threshold": self.acc_threshold
                },
                "gyroscope": {
                    "x": {"value": float(gyro_values[0]), "ratio": float(gyro_ratios[0])},
                    "y": {"value": float(gyro_values[1]), "ratio": float(gyro_ratios[1])},
                    "z": {"value": float(gyro_values[2]), "ratio": float(gyro_ratios[2])},
                    "threshold": self.gyro_threshold
                }
            }
            
            return is_anomaly, score, details
    
    def set_fallback_thresholds(self, acc_threshold: float = None, 
                              gyro_threshold: float = None):
        """
        Update the fallback detection thresholds.
        
        Args:
            acc_threshold: Acceleration threshold in g
            gyro_threshold: Gyroscope threshold in degrees/second
        """
        if acc_threshold is not None:
            self.acc_threshold = acc_threshold
        if gyro_threshold is not None:
            self.gyro_threshold = gyro_threshold
        
        logger.info(f"Updated fallback thresholds: acc={self.acc_threshold}g, "
                   f"gyro={self.gyro_threshold}deg/s")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = OneClassSVMDetector()
    
    # Example features: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    example_features = [0.1, 0.2, 1.1, 5.0, -2.0, 1.0]
    
    # Get prediction
    is_anomaly, score, details = detector.predict(example_features)
    
    print(f"Using fallback: {detector.using_fallback}")
    print(f"Anomaly detected: {is_anomaly}")
    print(f"Anomaly score: {score:.3f}")
    print(f"Threshold details: {details}")