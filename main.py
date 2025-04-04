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
    sensor = Sensor()
    
    # Initialize anomaly detector with SVM
    detector = OneClassSVMDetector(
        model_path='models/model_svm.pkl',
        sensitivity=args.sensitivity,
        threshold=args.threshold
    )
    
    # Initialize LCD display
    lcd = LCDDisplay()
    
    # Initialize SMS sender
    sms = SMSSender()
    
    # Main monitoring loop
    try:
        while True:
            # Read sensor data
            sensor_data = sensor.read_data()
            
            # Extract features
            features, feature_names = extract_features(sensor)
            
            # Check for anomalies
            if features is not None:
                print("Features extracted, running anomaly detection...")
                anomaly_score = detector.predict(features)
                is_anomaly = anomaly_score < args.threshold
                
                # Format alert message
                alert = format_alert(sensor_data=sensor_data)
                
                # Display on LCD
                lcd.display(alert)
                
                # Send SMS if anomaly detected and alerts are enabled
                if is_anomaly and alerts_enabled:
                    sms.send_alert(alert)
                    print("ANOMALY DETECTED!")
                else:
                    print("No anomaly detected")
            
            # Wait before next reading
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        lcd.clear()
        print("Monitoring system stopped")

if __name__ == "__main__":
    sys.exit(main())