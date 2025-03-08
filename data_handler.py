#!/usr/bin/env python3
"""
data_handler.py - Module for logging and analyzing acceleration data.

This module provides functions for:
- Recording sensor readings to CSV files
- Accumulating data over time windows
- Computing statistical features
- Detecting anomalies
- Logging results
"""

import csv
import os
import time
import statistics
from datetime import datetime
from collections import deque
from anomaly_detector import check_anomaly, DEFAULT_THRESHOLDS


def log_acceleration_to_csv(x, y, z, timestamp=None, filename='sensor_data.csv'):
    """
    Log acceleration data to a CSV file.
    
    Args:
        x (float): X-axis acceleration in m/s².
        y (float): Y-axis acceleration in m/s².
        z (float): Z-axis acceleration in m/s².
        timestamp: Timestamp for the reading. If None, current time is used.
                  Can be a datetime object or string.
        filename (str): Path to the CSV file. Default is 'sensor_data.csv'.
    
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    # If timestamp is not provided, use current time
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Prepare data row
    data_row = [timestamp, x, y, z]
    
    try:
        # Check if file exists to determine if header is needed
        file_exists = os.path.isfile(filename)
        
        # Open file in append mode
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header if file is being created
            if not file_exists:
                csv_writer.writerow(['timestamp', 'x', 'y', 'z'])
            
            # Write data row
            csv_writer.writerow(data_row)
        
        return True
    
    except IOError as e:
        print(f"Warning: Could not write to CSV file '{filename}': {e}")
        return False
    except Exception as e:
        print(f"Warning: An unexpected error occurred while writing to CSV: {e}")
        return False


def log_features_to_csv(features, is_anomaly, timestamp=None, filename='features_data.csv'):
    """
    Log calculated features and anomaly detection results to a CSV file.
    
    Args:
        features (dict): Dictionary of calculated features.
        is_anomaly (bool): Whether an anomaly was detected.
        timestamp: Timestamp for the features. If None, current time is used.
        filename (str): Path to the CSV file. Default is 'features_data.csv'.
    
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    # If timestamp is not provided, use current time
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    try:
        # Check if file exists to determine if header is needed
        file_exists = os.path.isfile(filename)
        
        # Prepare data row with timestamp, features, and anomaly flag
        data_row = [timestamp]
        
        # Prepare header if needed
        header = ['timestamp']
        
        # Add features in a consistent order
        for feature_name in sorted(features.keys()):
            header.append(feature_name)
            data_row.append(features[feature_name])
        
        # Add anomaly flag
        header.append('is_anomaly')
        data_row.append(1 if is_anomaly else 0)
        
        # Open file in append mode
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header if file is being created
            if not file_exists:
                csv_writer.writerow(header)
            
            # Write data row
            csv_writer.writerow(data_row)
        
        return True
    
    except IOError as e:
        print(f"Warning: Could not write to features CSV file '{filename}': {e}")
        return False
    except Exception as e:
        print(f"Warning: An unexpected error occurred while writing features to CSV: {e}")
        return False


def get_latest_readings(filename='sensor_data.csv', num_readings=1):
    """
    Retrieve the most recent acceleration readings from the CSV file.
    
    Args:
        filename (str): Path to the CSV file. Default is 'sensor_data.csv'.
        num_readings (int): Number of recent readings to retrieve.
    
    Returns:
        list: List of dictionaries containing the most recent readings.
              Returns an empty list if the file doesn't exist or on error.
    """
    try:
        if not os.path.isfile(filename):
            print(f"Warning: CSV file '{filename}' does not exist")
            return []
        
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            # Read header and all rows
            header = next(csv_reader)
            rows = list(csv_reader)
            
            # Get the most recent readings (last rows)
            recent_rows = rows[-num_readings:] if rows else []
            
            # Convert rows to dictionaries
            result = []
            for row in recent_rows:
                if len(row) >= 4:  # Ensure row has enough columns
                    result.append({
                        'timestamp': row[0],
                        'x': float(row[1]),
                        'y': float(row[2]),
                        'z': float(row[3])
                    })
            
            return result
    
    except IOError as e:
        print(f"Warning: Could not read from CSV file '{filename}': {e}")
        return []
    except Exception as e:
        print(f"Warning: An unexpected error occurred while reading CSV: {e}")
        return []


class AccelerationBuffer:
    """
    Buffer to accumulate acceleration data over time windows and compute features.
    """
    
    def __init__(self, window_size=1.0, expected_sample_rate=5):
        """
        Initialize the acceleration buffer.
        
        Args:
            window_size (float): Size of the time window in seconds. Default is 1.0.
            expected_sample_rate (int): Expected samples per second. Used to estimate buffer size.
        """
        self.window_size = window_size
        # Initialize deques with estimated capacity based on window size and sample rate
        self.buffer_capacity = int(window_size * expected_sample_rate * 1.5)  # 1.5x for safety
        self.timestamps = deque(maxlen=self.buffer_capacity)
        self.x_values = deque(maxlen=self.buffer_capacity)
        self.y_values = deque(maxlen=self.buffer_capacity)
        self.z_values = deque(maxlen=self.buffer_capacity)
        self.start_time = None
        self.features_filename = 'features_data.csv'
        self.thresholds = DEFAULT_THRESHOLDS
    
    def add_reading(self, x, y, z, timestamp=None):
        """
        Add an acceleration reading to the buffer.
        
        Args:
            x (float): X-axis acceleration.
            y (float): Y-axis acceleration.
            z (float): Z-axis acceleration.
            timestamp: Timestamp for the reading. If None, current time is used.
        
        Returns:
            bool: True if a window was completed and processed, False otherwise.
        """
        # If timestamp is not provided, use current time
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            # Parse ISO format timestamp string
            timestamp = datetime.fromisoformat(timestamp)
        
        # Initialize start_time if this is the first reading
        if self.start_time is None:
            self.start_time = timestamp
        
        # Add data to buffers
        self.timestamps.append(timestamp)
        self.x_values.append(x)
        self.y_values.append(y)
        self.z_values.append(z)
        
        # Check if we've accumulated enough data for a complete window
        window_complete = (timestamp - self.start_time).total_seconds() >= self.window_size
        
        if window_complete:
            # Process the window and reset
            self._process_window()
            # Update start time for next window
            self.start_time = timestamp
            return True
        
        return False
    
    def _process_window(self):
        """
        Process the current window of data: compute features, check for anomalies,
        and log the results.
        """
        if not self.x_values:  # Skip if buffer is empty
            return
        
        # Compute statistical features
        features = self._compute_features()
        
        # Check for anomalies
        is_anomaly, exceeded_features = check_anomaly(features, self.thresholds)
        
        # Get timestamp for this window (use the last timestamp)
        window_timestamp = self.timestamps[-1].isoformat()
        
        # Log features and anomaly results
        log_features_to_csv(features, is_anomaly, window_timestamp, self.features_filename)
        
        # Print message if anomaly detected
        if is_anomaly:
            print(f"ANOMALY DETECTED at {window_timestamp}:")
            for feature, (value, threshold) in exceeded_features.items():
                print(f"  {feature}: {value:.4f} exceeds threshold {threshold}")
    
    def _compute_features(self):
        """
        Compute statistical features from the buffered data.
        
        Returns:
            dict: Dictionary of computed features.
        """
        # Compute mean and standard deviation for each axis
        features = {
            "x_mean": statistics.mean(self.x_values),
            "y_mean": statistics.mean(self.y_values),
            "z_mean": statistics.mean(self.z_values),
            "x_std": statistics.stdev(self.x_values) if len(self.x_values) > 1 else 0,
            "y_std": statistics.stdev(self.y_values) if len(self.y_values) > 1 else 0,
            "z_std": statistics.stdev(self.z_values) if len(self.z_values) > 1 else 0
        }
        return features
    
    def set_thresholds(self, thresholds):
        """
        Set custom thresholds for anomaly detection.
        
        Args:
            thresholds (dict): Dictionary of feature names and threshold values.
        """
        self.thresholds = thresholds
    
    def set_features_filename(self, filename):
        """
        Set the filename for logging feature data.
        
        Args:
            filename (str): Path to the CSV file for features.
        """
        self.features_filename = filename
    
    def process_remaining_data(self):
        """
        Process any remaining data in the buffer (partial window).
        Call this before exiting the program to ensure all data is processed.
        """
        if self.x_values and len(self.x_values) >= 2:  # Need at least 2 points for std dev
            self._process_window()


# Example usage
if __name__ == "__main__":
    # Example of using the AccelerationBuffer
    buffer = AccelerationBuffer(window_size=1.0)
    
    # Simulate adding data over time
    for i in range(15):
        x = 0.1 + (i % 5) * 0.01  # Simulated x values
        y = 0.2 + (i % 3) * 0.02  # Simulated y values
        z = 9.8 + (i % 4) * 0.03  # Simulated z values
        
        timestamp = datetime.now()
        
        # Add reading to buffer
        window_processed = buffer.add_reading(x, y, z, timestamp)
        
        if window_processed:
            print(f"Processed window at {timestamp}")
        
        # Log raw data
        log_acceleration_to_csv(x, y, z, timestamp.isoformat())
        
        # Sleep for simulated time
        time.sleep(0.2)  # 5 Hz sample rate
    
    # Process any remaining data before exit
    buffer.process_remaining_data()
    print("Processed remaining data")