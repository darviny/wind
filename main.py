#!/usr/bin/env python3
import time
import sys
import board
import adafruit_mpu6050
import traceback
import numpy as np

from datetime import datetime
from lcd_alert import LCDAlert      

import sms_alert
import anomaly_detector
import sensor

def format_alert(anomaly_type=None, svm_score=None, confidence=None, sensor_data=None):
    alert = "================================================\n"
    alert += "WIND TURBINE ALERT\n"
    alert += "Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
    if anomaly_type:
        alert += "Type: " + anomaly_type + "\n"
    if svm_score is not None:
        alert += "SVM Score: " + str(svm_score) + "\n"
    if confidence is not None:
        alert += "Confidence: " + str(confidence) + "%\n"
    if sensor_data:
        alert += "Sensor Readings:\n"
        alert += "Accel (m/s²): X=" + "{:.2f}".format(sensor_data['accel_x']) + ", Y=" + "{:.2f}".format(sensor_data['accel_y']) + ", Z=" + "{:.2f}".format(sensor_data['accel_z']) + "\n"
        alert += "Gyro (deg/s): X=" + "{:.2f}".format(sensor_data['gyro_x']) + ", Y=" + "{:.2f}".format(sensor_data['gyro_y']) + ", Z=" + "{:.2f}".format(sensor_data['gyro_z']) + "\n"
        alert += "Temp: " + str(sensor_data['temp']) + "°C\n"
    alert += "================================================"
    return alert

def check_anomaly(model, buffer, svm_detector, rf_detector, sensor_data):
    if model == 'hybrid':
        # Get the latest window of data
        window = buffer.get_latest_window()
        if window is None:
            print("No window data available")
            return False
            
        # Use SVM detector to check for anomalies
        svm_score = svm_detector.predict(buffer.last_features)
        print(f"SVM score: {svm_score}")
        
        # If SVM detects an anomaly, use Random Forest to classify it
        if np.any(svm_score < 0):  # Negative score indicates anomaly
            print("SVM detected anomaly")
            # Use Random Forest to classify the anomaly type
            anomaly_type = rf_detector.predict(buffer.last_features)
            print(f"RF predicted anomaly type: {anomaly_type}")
            
            # Get probability estimates
            proba = rf_detector.model.predict_proba([buffer.last_features])[0]
            confidence = max(proba) * 100
            print(f"RF confidence: {confidence}%")
                
            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
            print(format_alert(anomaly_name, svm_score, confidence, sensor_data))
            return True
        else:
            print("SVM did not detect anomaly")
            
    elif model == 'rf':
        # Get the latest window of data
        window = buffer.get_latest_window()
        if window is None:
            print("No window data available")
            return False
            
        # Use Random Forest to classify the anomaly type
        anomaly_type = rf_detector.predict(buffer.last_features)
        print(f"RF predicted anomaly type: {anomaly_type}")
        
        # Get probability estimates
        proba = rf_detector.model.predict_proba([buffer.last_features])[0]
        confidence = max(proba) * 100
        print(f"RF confidence: {confidence}%")
        
        # If confidence is high enough, consider it an anomaly
        if confidence > 70:  # Threshold can be adjusted
            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
            print(format_alert(anomaly_name, confidence=confidence, sensor_data=sensor_data))
            return True
        else:
            print(f"RF confidence {confidence}% below threshold of 70%")
        
    else:  # svm mode
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
    if len(sys.argv) < 2:
        print("Usage: python main.py <model_type> <alerts_enabled> [sensitivity]")
        print("model_type: 'svm', 'rf', or 'hybrid'")
        print("alerts_enabled: 'true' or 'false'")
        print("sensitivity: float between 0.0 and 1.0 (default: 0.5)")
        return
        
    model = sys.argv[1].lower()
    alerts_enabled = sys.argv[2].lower() == 'true'
    sensitivity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    if model not in ['svm', 'rf', 'hybrid']:
        print("Invalid model type. Use 'svm', 'rf', or 'hybrid'")
        return
        
    print(f"Starting with {model} model, alerts {'enabled' if alerts_enabled else 'disabled'}, sensitivity {sensitivity}")
    
    try:
        print("Starting initialization...")
        
        # Initialize components
        if alerts_enabled:
            print("Initializing LCD...")
            try:
                lcd = LCDAlert()
                print("LCD initialized successfully")
                lcd.display_alert("Starting...")
                print("LCD test message displayed")
            except Exception as e:
                print(f"Error initializing LCD: {e}")
                lcd = None
        else:
            print("Alerts disabled, LCD not initialized")
            
        print("Initializing I2C...")
        i2c = board.I2C()
        print("I2C initialized successfully")
        
        print("Initializing MPU6050 sensor...")
        sensor_device = adafruit_mpu6050.MPU6050(i2c)
        print("MPU6050 sensor initialized successfully")
        
        print("Initializing sensor buffer...")
        buffer = sensor.SensorBuffer(window_size=1.0, expected_sample_rate=5)
        print("Sensor buffer initialized successfully")
        
        print("Loading anomaly detection models...")
        if model == 'hybrid':
            svm_detector = anomaly_detector.OneClassSVMDetector('models/model_svm.pkl', sensitivity=sensitivity)
            rf_detector = anomaly_detector.RandomForestDetector('models/model_rf.pkl')
        elif model == 'rf':
            svm_detector = None
            rf_detector = anomaly_detector.RandomForestDetector('models/model_rf.pkl')
        else:  # svm mode
            svm_detector = anomaly_detector.OneClassSVMDetector('models/model_svm.pkl', sensitivity=sensitivity)
            rf_detector = None
        print("Anomaly detection models loaded successfully")
        
        print("Components Ready")
        print(f"\nMonitoring started at 5 Hz with {model} model")
        print("Press Ctrl+C to stop")
        
        while True:
            # Read sensor
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
            
            sensor.log_sensor_data_to_csv(sensor_data, timestamp.isoformat())
            
            # Update display
            if lcd:
                lcd.lcd.clear()
                lcd.lcd.cursor_pos = (0, 0)
                lcd.lcd.write_string(f"X:{accel_x:.1f} Y:{accel_y:.1f}")
                lcd.lcd.cursor_pos = (1, 0)
                lcd.lcd.write_string(f"Z:{accel_z:.1f}")
            
            # Check for anomalies
            if buffer.add_reading(sensor_data, timestamp):
                print("Window complete, checking for anomalies...")
                features = anomaly_detector.extract_features(buffer)
                if features is not None:
                    print("Features extracted, running anomaly detection...")
                    is_anomaly = check_anomaly(model, buffer, svm_detector, rf_detector, sensor_data)
                    print(f"Anomaly detection result: {is_anomaly}")
                    
                    if is_anomaly:
                        print("ANOMALY DETECTED!")
                        if lcd:
                            lcd.display_alert("ANOMALY DETECTED!")
                            print("LCD updated with anomaly alert")
                        
                        if alerts_enabled:
                            alert_message = format_alert(sensor_data=sensor_data)
                            sms_alert.send_sms_alert('+17782383531', alert_message)
                            print("SMS alert sent")
            
            time.sleep(0.2)  # 5 Hz
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if buffer:
            buffer.process_remaining_data()

        
        print("\nMonitoring complete")
        return 0

if __name__ == "__main__":
    sys.exit(main())