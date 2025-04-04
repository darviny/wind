#!/usr/bin/env python3
import time
import sys
import board
import adafruit_mpu6050
import traceback
import numpy as np
import argparse

from datetime import datetime
from lcd_alert import LCDAlert      

import sms_alert
import anomaly_detector
import sensor

def format_alert(svm_score=None, sensor_data=None):
    alert = "================================================\n"
    alert += "WIND TURBINE ALERT\n"
    alert += "Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
    if svm_score is not None:
        alert += "SVM Score: " + str(svm_score) + "\n"
    if sensor_data:
        alert += "Sensor Readings:\n"
        alert += "Accel (m/s²): X=" + "{:.2f}".format(sensor_data['accel_x']) + ", Y=" + "{:.2f}".format(sensor_data['accel_y']) + ", Z=" + "{:.2f}".format(sensor_data['accel_z']) + "\n"
        alert += "Gyro (deg/s): X=" + "{:.2f}".format(sensor_data['gyro_x']) + ", Y=" + "{:.2f}".format(sensor_data['gyro_y']) + ", Z=" + "{:.2f}".format(sensor_data['gyro_z']) + "\n"
        alert += "Temp: " + str(sensor_data['temp']) + "°C\n"
    alert += "================================================"
    return alert

def check_anomaly(buffer, svm_detector, sensor_data):
    # Get the latest window of data
    window = buffer.get_latest_window()
    if window is None:
        print("No window data available")
        return False
        
    # Use SVM detector to check for anomalies
    svm_score = svm_detector.predict(buffer.last_features)
    print(f"SVM score: {svm_score}")
    
    # If SVM detects an anomaly
    if np.any(svm_score < 0):  # Negative score indicates anomaly
        print("SVM detected anomaly")
        print(format_alert(svm_score=svm_score, sensor_data=sensor_data))
        return True
    else:
        print("SVM did not detect anomaly")
    
    return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wind Turbine Monitoring System')
    parser.add_argument('alerts_enabled', type=str, help='Enable alerts (true/false)')
    parser.add_argument('sensitivity', type=float, help='SVM sensitivity (0.0 to 1.0)')
    parser.add_argument('--threshold', type=float, default=-1.0, help='Anomaly detection threshold (default: -1.0)')
    args = parser.parse_args()
    
    # Convert alerts_enabled to boolean
    alerts_enabled = args.alerts_enabled.lower() == 'true'
    
    # Initialize sensor
    sensor_buffer = sensor.SensorBuffer(window_size=1.0, expected_sample_rate=5)
    
    # Initialize anomaly detector with SVM
    detector = anomaly_detector.OneClassSVMDetector(
        model_path='models/model_svm.pkl',
        sensitivity=args.sensitivity,
        threshold=args.threshold
    )
    
    # Initialize LCD display
    lcd = LCDAlert()
    
    # Initialize SMS sender
    sms = sms_alert.SMSAlert()
    
    # Main monitoring loop
    try:
        while True:
            # Read sensor data
            accel = sensor_device.acceleration
            gyro = sensor_device.gyro
            temp = sensor_device.temperature
            timestamp = datetime.now()
            accel_x, accel_y, accel_z = accel
            gyro_x, gyro_y, gyro_z = gyro
            
            sensor_data = {
                'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
                'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
                'temp': temp
            }
            
            # Extract features
            features, feature_names = anomaly_detector.extract_features(sensor_buffer)
            
            # Check for anomalies
            if features is not None:
                print("Features extracted, running anomaly detection...")
                anomaly_score = detector.predict(features)
                is_anomaly = anomaly_score < args.threshold
                
                # Format alert message
                alert = format_alert(anomaly_score, sensor_data)
                
                # Display on LCD
                lcd.display_alert(alert)
                
                # Send SMS if anomaly detected and alerts are enabled
                if is_anomaly and alerts_enabled:
                    sms.send_sms_alert('+17782383531', alert)
                    print("ANOMALY DETECTED!")
                else:
                    print("No anomaly detected")
            
            # Wait before next reading
            time.sleep(0.2)  # 5 Hz
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        lcd.clear()
        print("Monitoring system stopped")

if __name__ == "__main__":
    sys.exit(main())