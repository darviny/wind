#!/usr/bin/env python3
"""
main.py - Main script for MPU6050 sensor monitoring system.

This script integrates all components of the monitoring system:
- Sensor reading from MPU6050 accelerometer
- Data buffering and feature calculation
- Anomaly detection
- Alerting

It can be run directly or integrated into a systemd service.
"""

import time
import signal
import sys
import board
import adafruit_mpu6050
from datetime import datetime
from data_handler import AccelerationBuffer, log_sensor_data_to_csv

# Import anomaly detection
from anomaly_detector import DEFAULT_THRESHOLDS, check_anomaly

# Import alerting
from alert import send_alert, log_alert

# Global flag for clean exit
running = True

def signal_handler(sig, frame):
    """Handle clean exit when Ctrl+C is pressed"""
    global running
    print("\nStopping data collection...")
    running = False

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    sensor = None
    buffer = None
    
    try:
        # Initialize I2C and MPU6050 sensor
        print("Initializing MPU6050 sensor...")
        i2c = board.I2C()  # uses board.SCL and board.SDA
        sensor = adafruit_mpu6050.MPU6050(i2c)
        print("Sensor initialized successfully")
        
        # Create data buffer for 1-second windows
        buffer = AccelerationBuffer(window_size=1.0, expected_sample_rate=5)
        
        print("Starting data collection at 5 Hz. Press Ctrl+C to exit.")
        
        # Main data collection loop - runs until Ctrl+C is pressed
        while running:
            try:
                # Read all sensor data
                accel = sensor.acceleration  # This returns a tuple (x, y, z)
                gyro = sensor.gyro          # This returns a tuple (x, y, z)
                temp = sensor.temperature   # This returns a single value
                timestamp = datetime.now()
                
                # Unpack the tuples before logging
                accel_x, accel_y, accel_z = accel  # Unpack acceleration tuple
                gyro_x, gyro_y, gyro_z = gyro      # Unpack gyroscope tuple
                
                # Log raw data to CSV with unpacked values
                success = log_sensor_data_to_csv(
                    accel_x, accel_y, accel_z,     # Unpacked acceleration values
                    gyro_x, gyro_y, gyro_z,        # Unpacked gyroscope values
                    temp,                           # Temperature (already a single value)
                    timestamp.isoformat()
                )
                
                if success:
                    # Print the current readings using unpacked values
                    print(f"\rAccel: ({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f}) m/s² | "
                          f"Gyro: ({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f}) rad/s | "
                          f"Temp: {temp:.1f}°C", end='')
                
                # Add data to the processing buffer (use individual values, not tuples)
                window_processed = buffer.add_reading(
                    accel_x, accel_y, accel_z,  # individual acceleration values
                    timestamp
                )
                
                if window_processed:
                    # The anomaly message will be printed automatically by _process_window
                    # if an anomaly is detected
                    print("\nProcessed 1-second window of data")
                
                # Wait for next sample (5 Hz = 0.2 seconds)
                time.sleep(0.2)
                
            except Exception as e:
                print(f"\nError reading sensor: {e}")
                time.sleep(0.2)  # Wait before retry
    
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check:")
        print("1. Sensor connections (SDA, SCL, VCC, GND)")
        print("2. I2C is enabled (sudo raspi-config)")
        print("3. I2C permissions (sudo usermod -aG i2c $USER)")
        return 1
    finally:
        # Clean up
        if buffer:
            print("Processing remaining data...")
            buffer.process_remaining_data()
    
    print("Data collection complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())