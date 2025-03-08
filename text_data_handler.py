#!/usr/bin/env python3
"""
test_data_handler.py - Test script for sensor_reader.py and data_handler.py.

This script continuously reads accelerometer, gyroscope, and temperature data 
from an MPU6050 sensor and logs it to a CSV file at 5 Hz (every 0.2 seconds).
"""

import time
import signal
import sys
from datetime import datetime
from sensor_reader import MPU6050Reader
from data_handler import log_sensor_data_to_csv

# Global flag for clean exit
running = True

# Set up clean exit handling
def signal_handler(sig, frame):
    global running
    print("\nStopping data collection...")
    running = False

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    sensor = None
    
    try:
        # Initialize the MPU6050 sensor
        print("Initializing MPU6050 sensor...")
        sensor = MPU6050Reader()
        print("Sensor initialized successfully")
        
        print("Starting data logging at 5 Hz. Press Ctrl+C to exit.")
        
        # Main loop for reading and logging data
        while running:
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Read sensor data
            accel = sensor.get_acceleration()
            gyro = sensor.get_gyro()
            temp = sensor.get_temperature()
            
            # Log data to CSV
            if log_sensor_data_to_csv(
                accel[0], accel[1], accel[2],  # acceleration x, y, z
                gyro[0], gyro[1], gyro[2],     # gyroscope x, y, z
                temp,                           # temperature
                timestamp
            ):
                print(f"Logged: Accel(x={accel[0]:.2f}, y={accel[1]:.2f}, z={accel[2]:.2f}), "
                      f"Gyro(x={gyro[0]:.2f}, y={gyro[1]:.2f}, z={gyro[2]:.2f}), "
                      f"Temp={temp:.1f}°C")
            else:
                print("Failed to log data")
            
            # Wait for next reading (5 Hz = 0.2 seconds)
            time.sleep(0.2)
        
        print("Data collection complete.")
    
    except KeyboardInterrupt:
        # Extra handling for Ctrl+C in case signal handler doesn't catch it
        print("\nData collection interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up if needed
        if sensor:
            sensor.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())