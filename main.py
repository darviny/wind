#!/usr/bin/env python3
"""
main.py - Main script for MPU6050 sensor monitoring system.

This script integrates all components of the monitoring system:
- Sensor reading from MPU6050 accelerometer
- Data buffering and feature calculation
- Anomaly detection
- LCD Alerting

It can be run directly or integrated into a systemd service.
"""

import time
import signal
import sys
import board
import adafruit_mpu6050
from datetime import datetime
from data_handler import AccelerationBuffer, log_sensor_data_to_csv
from lcd_alert import LCDAlert
from sms_alert import send_sms_alert

# Import anomaly detection
from anomaly_detector import DEFAULT_THRESHOLDS, check_anomaly

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
    lcd = None
    
    try:
        # Initialize LCD
        print("Initializing LCD...")
        try:
            lcd = LCDAlert()
            lcd.display_alert("Starting up...")
        except Exception as e:
            print(f"Warning: LCD initialization failed: {e}")
            lcd = None

        # Initialize I2C and MPU6050 sensor
        print("Initializing MPU6050 sensor...")
        i2c = board.I2C()  # uses board.SCL and board.SDA
        sensor = adafruit_mpu6050.MPU6050(i2c)
        print("Sensor initialized successfully")
        if lcd:
            lcd.display_alert("Sensor Ready", duration=1)
        
        # Create data buffer for 1-second windows
        buffer = AccelerationBuffer(window_size=1.0, expected_sample_rate=5)
        
        print("Starting data collection at 5 Hz. Press Ctrl+C to exit.")
        if lcd:
            lcd.display_alert("Monitoring...")
        
        # Main data collection loop - runs until Ctrl+C is pressed
        while running:
            try:
                # Read all sensor data
                accel = sensor.acceleration
                gyro = sensor.gyro
                temp = sensor.temperature
                timestamp = datetime.now()
                
                # Unpack the tuples before logging
                accel_x, accel_y, accel_z = accel
                gyro_x, gyro_y, gyro_z = gyro
                
                # Log raw data to CSV with unpacked values
                success = log_sensor_data_to_csv(
                    accel_x, accel_y, accel_z,
                    gyro_x, gyro_y, gyro_z,
                    temp,
                    timestamp.isoformat()
                )
                
                if success and lcd:
                    # Update LCD with current readings
                    lcd.lcd.clear()
                    lcd.lcd.cursor_pos = (0, 0)
                    lcd.lcd.write_string(f"X:{accel_x:.1f} Y:{accel_y:.1f}")
                    lcd.lcd.cursor_pos = (1, 0)
                    lcd.lcd.write_string(f"Z:{accel_z:.1f}")
                
                # Add data to the processing buffer
                window_processed = buffer.add_reading(
                    accel_x, accel_y, accel_z,
                    timestamp
                )
                
                if window_processed:
                    # Get features and check for anomalies
                    features = buffer._compute_features()
                    is_anomaly, exceeded = check_anomaly(features, DEFAULT_THRESHOLDS)
                    
                    if is_anomaly:
                        print("\nANOMALY DETECTED!")
                        if lcd: 
                            lcd.display_alert("ANOMALY DETECTED!", duration=1)
                            send_sms_alert("+17782383531", "Anomaly detected!")
                    else:
                        print("\nProcessed 1-second window of data")
                
                # Wait for next sample (5 Hz = 0.2 seconds)
                time.sleep(0.2)
                
            except Exception as e:
                print(f"\nError reading sensor: {e}")
                if lcd:
                    lcd.display_alert(f"Error: {str(e)[:16]}")
                time.sleep(0.2)
    
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
        if lcd:
            lcd.display_alert("Stopping...", duration=1)
    except Exception as e:
        print(f"Error: {e}")
        if lcd:
            lcd.display_alert("Error! Check log")
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
        if lcd:
            lcd.clear()
    
    print("Data collection complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())