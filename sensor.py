from datetime import datetime
import numpy as np
import csv
import os
from helpers.extract_acf_features import compute_acf_features, compute_aggregate_features


def log_sensor_data_to_csv(sensor_data, timestamp=None, filename='data/sensor_data.csv'):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    data_row = [timestamp, 
                sensor_data['accel_x'], sensor_data['accel_y'], sensor_data['accel_z'],
                sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z'],
                sensor_data['temp']]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['timestamp', 
                               'accel_x', 'accel_y', 'accel_z',
                               'gyro_x', 'gyro_y', 'gyro_z',
                               'temperature'])
        csv_writer.writerow(data_row)
    return True


class SensorBuffer:
    def __init__(self, window_size, expected_sample_rate=5):
        self.window_size = window_size
        self.samples_needed = int(window_size * expected_sample_rate)
        self.start_time = None
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []

    def add_reading(self, sensor_data, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        # Add new reading to buffer
        self.accel_x.append(sensor_data['accel_x'])
        self.accel_y.append(sensor_data['accel_y'])
        self.accel_z.append(sensor_data['accel_z'])
        self.gyro_x.append(sensor_data['gyro_x'])
        self.gyro_y.append(sensor_data['gyro_y'])
        self.gyro_z.append(sensor_data['gyro_z'])
        
        window_complete = (timestamp - self.start_time).total_seconds() >= self.window_size
        
        if window_complete and len(self.accel_x) >= self.samples_needed:
            self._process_window()
            self.start_time = timestamp
            return True
        
        return False

    def _process_window(self):
        if len(self.accel_x) < self.samples_needed:
            return None
            
        # Create numpy array from sensor data
        # Shape before transpose: (6, n_samples)
        # - 6 rows (one for each sensor: accel_x,y,z and gyro_x,y,z)
        # - n_samples columns (one for each reading)
        window = np.array([
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z
        ])
        
        # Transpose to shape (n_samples, 6)
        # - n_samples rows (one for each reading)
        # - 6 columns (one for each sensor)
        window = window.T
        
        # Initialize features list
        features = []
        
        # For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        for i in range(6):  # 6 sensors
            sensor_data = window[:, i]
            
            # Add statistical features
            features.extend(compute_aggregate_features(sensor_data))
            
            # Add ACF features
            features.extend(compute_acf_features(sensor_data))
        
        # Reset buffer
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        
        return np.array(features)
            
    def process_remaining_data(self):
        if self.accel_x:
            return self._process_window()
        return None

    def get_latest_window(self):
        print(f"Buffer lengths: accel_x={len(self.accel_x)}, accel_y={len(self.accel_y)}, accel_z={len(self.accel_z)}, "
              f"gyro_x={len(self.gyro_x)}, gyro_y={len(self.gyro_y)}, gyro_z={len(self.gyro_z)}")
        
        if len(self.accel_x) < self.samples_needed:
            print(f"Not enough samples. Need {self.samples_needed}, have {len(self.accel_x)}")
            return None
            
        # Create numpy array from sensor data
        # Shape before transpose: (6, n_samples)
        # - 6 rows (one for each sensor: accel_x,y,z and gyro_x,y,z)
        # - n_samples columns (one for each reading)
        window = np.array([
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z
        ])
        
        # Transpose to shape (n_samples, 6)
        # - n_samples rows (one for each reading)
        # - 6 columns (one for each sensor)
        window = window.T
        print(f"Window shape: {window.shape}")
        return window