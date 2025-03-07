#!/usr/bin/env python3
"""
data_handler.py - Module for logging acceleration data to CSV files.

This module provides functions for recording sensor readings to CSV files
with appropriate error handling and file management.
"""

import csv
import os
from datetime import datetime


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


# Example usage
if __name__ == "__main__":
    # Example of logging current acceleration data
    x, y, z = 0.1, 0.2, 0.9  # Example acceleration values
    success = log_acceleration_to_csv(x, y, z)
    
    if success:
        print(f"Data logged successfully to sensor_data.csv")
        
        # Read back the latest data
        latest = get_latest_readings()
        if latest:
            print(f"Latest reading: {latest[0]}")
    else:
        print("Failed to log data")