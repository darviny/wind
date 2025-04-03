#!/usr/bin/env python3
import time
import sys
import board
import adafruit_mpu6050
import traceback

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
        # Extract features using the function from anomaly_detector.py
        features = anomaly_detector.extract_features(buffer)
        if features is None:
            return False
            
        # Use SVM detector to check for anomalies
        svm_score = svm_detector.predict(features)
        
        # SVM Score Explanation:
        # - Positive scores: Sample is likely "normal" (inside decision boundary)
        # - Negative scores: Sample is likely an anomaly (outside decision boundary)
        # - Score magnitude: Further from zero = more confident prediction
        #   * Large positive: Very confident sample is normal
        #   * Large negative: Very confident sample is an anomaly
        #   * Near zero: Sample is near decision boundary (uncertain)
        # - Threshold of 0 is used to classify samples as normal or anomalous
        
        # If SVM detects an anomaly, use Random Forest to classify it
        if svm_score < 0:  # Negative score indicates anomaly
            # Use Random Forest to classify the anomaly type
            anomaly_type = rf_detector.predict(features)
            
            # Get probability estimates
            proba = rf_detector.model.predict_proba([features])[0]
            confidence = max(proba) * 100
                
            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
            print(format_alert(anomaly_name, svm_score, confidence, sensor_data))
            return True
            
    elif model == 'rf':
        # Extract features using the function from anomaly_detector.py
        features = anomaly_detector.extract_features(buffer)
        if features is None:
            return False
            
        # Use Random Forest to classify the anomaly type
        anomaly_type = rf_detector.predict(features)
        
        # Get probability estimates
        proba = rf_detector.model.predict_proba([features])[0]
        confidence = max(proba) * 100
        
        # If confidence is high enough, consider it an anomaly
        if confidence > 70:  # Threshold can be adjusted
            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
            print(format_alert(anomaly_name, confidence=confidence, sensor_data=sensor_data))
            return True
        
    else:  # svm mode
        # Extract features using the function from anomaly_detector.py
        features = anomaly_detector.extract_features(buffer)
        if features is None:
            return False
            
        # Use SVM detector to check for anomalies
        svm_score = svm_detector.predict(features)
        
        # SVM Score Explanation:
        # - Positive scores: Sample is likely "normal" (inside decision boundary)
        # - Negative scores: Sample is likely an anomaly (outside decision boundary)
        # - Score magnitude: Further from zero = more confident prediction
        #   * Large positive: Very confident sample is normal
        #   * Large negative: Very confident sample is an anomaly
        #   * Near zero: Sample is near decision boundary (uncertain)
        # - Threshold of 0 is used to classify samples as normal or anomalous
        
        # If SVM detects an anomaly
        if svm_score < 0:  # Negative score indicates anomaly
            print(format_alert(svm_score=svm_score, sensor_data=sensor_data))
            return True
    
    return False

def main():
    # Arguments
    model = 'svm'
    alerts_enabled = False
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 2:
        alerts_enabled = sys.argv[2].lower() == 'true'
    
    lcd = None
    buffer = None
    
    try:
        print("Starting initialization...")
        
        # Initialize components
        if alerts_enabled:
            print("Initializing LCD...")
            lcd = LCDAlert()
            lcd.display_alert("Starting...")
            sms_alert.set_cooldown_period(5)
            print("LCD initialized successfully")
     
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
            svm_detector = anomaly_detector.OneClassSVMDetector('models/model_svm.pkl')
            rf_detector = anomaly_detector.RandomForestDetector('models/model_rf.pkl')
        elif model == 'rf':
            svm_detector = None
            rf_detector = anomaly_detector.RandomForestDetector('models/model_rf.pkl')
        else:  # svm mode
            svm_detector = anomaly_detector.OneClassSVMDetector('models/model_svm.pkl')
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
                is_anomaly = check_anomaly(model, buffer, svm_detector, rf_detector, sensor_data)
                
                if is_anomaly:
                    if lcd:
                        lcd.display_alert("ANOMALY DETECTED!")
                    
                    if alerts_enabled:
                        alert_message = format_alert(sensor_data=sensor_data)
                        sms_alert.send_sms_alert('+17782383531', alert_message)
            
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