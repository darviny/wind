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
                accel = sensor.acceleration
                gyro = sensor.gyro
                temp = sensor.temperature
                timestamp = datetime.now()
                
                # Log raw data to CSV
                success = log_sensor_data_to_csv(
                    accel[0], accel[1], accel[2],  # acceleration
                    gyro[0], gyro[1], gyro[2],     # gyroscope
                    temp,                           # temperature
                    timestamp.isoformat()
                )
                
                if success:
                    # Print the current readings
                    print(f"\rAccel: ({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}) m/s² | "
                          f"Gyro: ({gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f}) rad/s | "
                          f"Temp: {temp:.1f}°C", end='')
                
                # Add data to the processing buffer
                window_processed = buffer.add_reading(accel, gyro, temp, timestamp)
                
                if window_processed:
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