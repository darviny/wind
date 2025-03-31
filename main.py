#!/usr/bin/env python3
"""
main.py - Main script for MPU6050 sensor monitoring system.

Usage:
    python main.py                     # Run with all alerts enabled
    python main.py --no-sms           # Run without SMS alerts
    python main.py train              # Run in training mode
    python main.py train --no-sms     # Run in training mode without SMS
"""

import time
import signal
import sys
import board
import adafruit_mpu6050
import argparse
from datetime import datetime
from data_handler import AccelerationBuffer, log_sensor_data_to_csv
from lcd_alert import LCDAlert
from sms_alert import send_sms_alert, set_cooldown_period
from anomaly_detector import OneClassSVMDetector

# Global flag for clean exit
running = True

def signal_handler(sig, frame):
    """Handle clean exit when Ctrl+C is pressed"""
    global running
    print("\nStopping data collection...")
    running = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MPU6050 sensor monitoring system')
    parser.add_argument('mode', nargs='?', default='detect',
                      choices=['detect', 'train'],
                      help='Operating mode: detect (default) or train')
    parser.add_argument('--no-sms', action='store_true',
                      help='Disable SMS alerts')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    training_mode = args.mode == 'train'
    sms_enabled = not args.no_sms
    
    if not sms_enabled:
        print("SMS alerts are disabled")
    
    sensor = None
    buffer = None
    lcd = None
    
    try:
        # Initialize LCD
        print("Initializing LCD...")
        try:
            lcd = LCDAlert()
            if training_mode:
                lcd.display_alert("Training Mode...")
            else:
                lcd.display_alert("Starting up...")
        except Exception as e:
            print(f"Warning: LCD initialization failed: {e}")
            lcd = None

        # Initialize I2C and MPU6050 sensor
        print("Initializing MPU6050 sensor...")
        i2c = board.I2C()
        sensor = adafruit_mpu6050.MPU6050(i2c)
        print("Sensor initialized successfully")
        if lcd:
            lcd.display_alert("Sensor Ready", duration=1)
        
        # Create data buffer for 1-second windows
        buffer = AccelerationBuffer(window_size=1.0, expected_sample_rate=5)
        
        if training_mode:
            print("Starting data collection in TRAINING MODE at 5 Hz. Press Ctrl+C to exit.")
            if lcd:
                lcd.display_alert("Training Mode")
        else:
            print("Starting data collection at 5 Hz. Press Ctrl+C to exit.")
            if lcd:
                lcd.display_alert("Monitoring...")
            
            # Initialize detector
            detector = OneClassSVMDetector()
            if detector.using_fallback:
                detector.set_fallback_thresholds(acc_threshold=2.5, gyro_threshold=150.0)
            
            # Only set cooldown if SMS is enabled
            if sms_enabled:
                set_cooldown_period(5)
        
        # Main data collection loop
        while running:
            try:
                # Read all sensor data
                accel = sensor.acceleration
                gyro = sensor.gyro
                temp = sensor.temperature
                timestamp = datetime.now()
                
                # Unpack the tuples
                accel_x, accel_y, accel_z = accel
                gyro_x, gyro_y, gyro_z = gyro
                
                # Log raw data to CSV
                filename = 'training_data.csv' if training_mode else 'sensor_data.csv'
                success = log_sensor_data_to_csv(
                    accel_x, accel_y, accel_z,
                    gyro_x, gyro_y, gyro_z,
                    temp,
                    timestamp.isoformat(),
                    filename
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
                
                if window_processed and not training_mode:
                    # Get features and check for anomalies
                    features = buffer._compute_features()
                    feature_values = [
                        features['accel_x_mean'], features['accel_y_mean'], features['accel_z_mean'],
                        features['accel_x_std'], features['accel_y_std'], features['accel_z_std'],
                        features['accel_x_max'], features['accel_y_max'], features['accel_z_max']
                    ]
                    is_anomaly, score = detector.predict(feature_values)
                    
                    # Print current readings
                    print("\n=== Current Readings ===")
                    print(f"Acceleration (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                    print(f"Gyroscope (deg/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                    print(f"Temperature: {temp:.1f}°C")
                    print(f"Anomaly Score: {score:.3f}")
                    
                    if is_anomaly:
                        print("*** ANOMALY DETECTED! ***")
                        if lcd:
                            lcd.display_alert("ANOMALY DETECTED!", duration=1)
                        if sms_enabled:
                            send_sms_alert(
                                "+17782383531",
                                f"Anomaly detected! Values: X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}"
                            )
                    print("=" * 23)
                
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