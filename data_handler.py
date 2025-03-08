#!/usr/bin/env python3
"""
data_handler.py - Module for logging MPU6050 sensor data to CSV files.

This module provides functions for recording accelerometer, gyroscope,
and temperature readings to CSV files with appropriate error handling.
"""

import csv
import os
from datetime import datetime


def log_sensor_data_to_csv(accel_x, accel_y, accel_z, 
                          gyro_x, gyro_y, gyro_z,
                          temperature,
                          timestamp=None, 
                          filename='sensor_data.csv'):
    """
    Log MPU6050 sensor data to a CSV file.
    
    Args:
        accel_x, accel_y, accel_z (float): Acceleration values in m/s².
        gyro_x, gyro_y, gyro_z (float): Gyroscope values in rad/s.
        temperature (float): Temperature in degrees Celsius.
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
    data_row = [timestamp, 
                accel_x, accel_y, accel_z,
                gyro_x, gyro_y, gyro_z,
                temperature]
    
    try:
        # Check if file exists to determine if header is needed
        file_exists = os.path.isfile(filename)
        
        # Open file in append mode
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header if file is being created
            if not file_exists:
                csv_writer.writerow(['timestamp', 
                                   'accel_x', 'accel_y', 'accel_z',
                                   'gyro_x', 'gyro_y', 'gyro_z',
                                   'temperature'])
            
            # Write data row
            csv_writer.writerow(data_row)
        
        return True
    
    except IOError as e:
        print(f"Warning: Could not write to CSV file '{filename}': {e}")
        return False
    except Exception as e:
        print(f"Warning: An unexpected error occurred while writing to CSV: {e}")
        return False


def get_latest_readings(filename='sensor_data.csv', num_readings=1):
    """
    Retrieve the most recent sensor readings from the CSV file.
    
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
                if len(row) >= 8:  # Ensure row has enough columns
                    result.append({
                        'timestamp': row[0],
                        'acceleration': {
                            'x': float(row[1]),
                            'y': float(row[2]),
                            'z': float(row[3])
                        },
                        'gyro': {
                            'x': float(row[4]),
                            'y': float(row[5]),
                            'z': float(row[6])
                        },
                        'temperature': float(row[7])
                    })
            
            return result
    
    except IOError as e:
        print(f"Warning: Could not read from CSV file '{filename}': {e}")
        return []
    except Exception as e:
        print(f"Warning: An unexpected error occurred while reading CSV: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Example of logging current sensor data
    accel = (0.1, 0.2, 0.9)  # Example acceleration values
    gyro = (0.01, 0.02, 0.03)  # Example gyroscope values
    temp = 25.5  # Example temperature value
    
    success = log_sensor_data_to_csv(
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2],
        temp
    )
    
    if success:
        print(f"Data logged successfully to sensor_data.csv")
        
        # Read back the latest data
        latest = get_latest_readings()
        if latest:
            print(f"Latest reading: {latest[0]}")
    else:
        print("Failed to log data")