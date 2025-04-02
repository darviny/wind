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
import joblib

# Global flag for clean exit
running = True

def signal_handler(sig, frame):
    """Handle clean exit when Ctrl+C is pressed"""
    global running
    print("\nStopping data collection...")
    running = False

# Add near the start of the script, after the signal handler definition
signal.signal(signal.SIGINT, signal_handler)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MPU6050 sensor monitoring system')
    parser.add_argument('mode', nargs='?', default='detect',
                      choices=['detect', 'train'],
                      help='Operating mode: detect (default) or train')
    parser.add_argument('--no-sms', action='store_true',
                      help='Disable SMS alerts')
    parser.add_argument('--model-type', choices=['rf', 'svm', 'hybrid'], default='rf',
                      help='Model type to use: rf (Random Forest), svm (One-Class SVM), or hybrid')
    parser.add_argument('--svm-model', default='svm_model.pkl',
                      help='Path to SVM model file (for svm or hybrid mode)')
    parser.add_argument('--rf-model', default='rf_model.pkl',
                      help='Path to Random Forest model file (for rf or hybrid mode)')
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
        buffer = SensorBuffer(window_size=1.0, expected_sample_rate=5)
        
        if training_mode:
            print("Starting data collection in TRAINING MODE at 5 Hz. Press Ctrl+C to exit.")
            if lcd:
                lcd.display_alert("Training Mode")
        else:
            print("Starting data collection at 5 Hz. Press Ctrl+C to exit.")
            if lcd:
                lcd.display_alert("Monitoring...")
            
            # Initialize appropriate detector(s)
            if args.model_type == 'hybrid':
                print("Initializing hybrid detection mode...")
                svm_detector = OneClassSVMDetector(args.svm_model)
                rf_detector = TurbineAnomalyDetector(args.rf_model)
                print("Both models loaded successfully")
            elif args.model_type == 'rf':
                rf_detector = TurbineAnomalyDetector(args.rf_model)
            else:  # svm mode
                svm_detector = OneClassSVMDetector(args.svm_model)
            
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
                
                # Log raw data to CSV with unpacked values
                success = log_sensor_data_to_csv(
                    accel_x, accel_y, accel_z,
                    gyro_x, gyro_y, gyro_z,
                    temp,
                    timestamp.isoformat()
                )
                
                # Debug: Print values being logged
                print("\n=== Data Being Logged ===")
                print(f"Time: {timestamp.isoformat()}")
                print(f"Acceleration (m/s²): X={accel_x:.3f}, Y={accel_y:.3f}, Z={accel_z:.3f}")
                print(f"Gyroscope (deg/s): X={gyro_x:.3f}, Y={gyro_y:.3f}, Z={gyro_z:.3f}")
                print(f"Temperature: {temp:.1f}°C")
                print(f"Log success: {success}")
                print("=" * 20)
                
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
                    gyro_x, gyro_y, gyro_z,
                    timestamp
                )
                
                if window_processed and not training_mode:
                    # Extract features and make prediction
                    if args.model_type == 'hybrid':
                        svm_features = svm_detector.extract_features(buffer)
                    elif args.model_type == 'rf':
                        features = rf_detector.extract_features(buffer)
                    else:  # svm mode
                        features = svm_detector.extract_features(buffer)
                    
                    # Print current readings first
                    print("\n=== Current Readings ===")
                    print(f"Time: {timestamp.isoformat()}")
                    print(f"Model: {args.model_type.upper()}")
                    print(f"Acceleration (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                    print(f"Gyroscope (deg/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                    print(f"Temperature: {temp:.1f}°C")
                    
                    if args.model_type == 'hybrid':
                        print("\n=== Hybrid Detection Results ===")
                        # Extract features for SVM
                        svm_features = svm_detector.extract_features(buffer)
                        is_anomaly, svm_score = svm_detector.predict(svm_features)
                        
                        # Log SVM results
                        print(f"SVM Analysis:")
                        print(f"└── Decision Score: {svm_score:.3f}")
                        print(f"└── Initial Detection: {'Anomaly' if is_anomaly else 'Normal'}")
                        
                        if is_anomaly:
                            # If SVM detects anomaly, use RF to classify it
                            print("\nActivating Random Forest for classification...")
                            rf_features = rf_detector.extract_features(buffer)
                            rf_is_anomaly, anomaly_type = rf_detector.predict(rf_features)
                            
                            # Get probability scores if available
                            try:
                                proba = rf_detector.model.predict_proba([rf_features])[0]
                                class_confidence = max(proba) * 100
                            except:
                                class_confidence = None
                            
                            # Log RF results
                            print(f"Random Forest Analysis:")
                            print(f"└── Classification: Type {anomaly_type}")
                            if class_confidence is not None:
                                print(f"└── Confidence: {class_confidence:.1f}%")
                            
                            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
                            alert_msg = (
                                f"Anomaly detected!\n"
                                f"Type: {anomaly_name}\n"
                                f"SVM Score: {svm_score:.3f}"
                                + (f"\nConfidence: {class_confidence:.1f}%" if class_confidence is not None else "")
                            )
                            
                            print("\nFinal Decision:")
                            print(f"└── {alert_msg.replace('\n', '\n    ')}")
                        
                    elif args.model_type == 'rf':
                        # Extract features and predict
                        features = rf_detector.extract_features(buffer)
                        is_anomaly, anomaly_type = rf_detector.predict(features)
                        
                        if is_anomaly:
                            anomaly_name = "Tempered Blade" if anomaly_type == 1 else "Gearbox Issue"
                            alert_msg = f"Anomaly detected! Type: {anomaly_name}"
                    
                    else:  # svm mode
                        # Extract features and predict
                        features = svm_detector.extract_features(buffer)
                        is_anomaly, svm_score = svm_detector.predict(features)
                        
                        if is_anomaly:
                            alert_msg = f"Anomaly detected! Score: {svm_score:.3f}"
                    
                    if is_anomaly:
                        print("\n" + "!" * 50)
                        print(f"*** {alert_msg} ***")
                        print("!" * 50)
                        
                        if lcd:
                            # For hybrid mode, show more detailed alert on LCD
                            if args.model_type == 'hybrid':
                                if 'anomaly_name' in locals():  # Check if variable exists
                                    lcd.display_alert(f"ANOMALY: {anomaly_name}")
                                else:
                                    lcd.display_alert("ANOMALY DETECTED!")
                            else:
                                lcd.display_alert("ANOMALY DETECTED!")
                        
                        if sms_enabled:
                            # Format SMS message based on model type
                            if args.model_type == 'hybrid':
                                sms_msg = (
                                    f"WIND TURBINE ALERT\n"
                                    f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Type: {anomaly_name}\n"
                                    f"SVM Score: {svm_score:.3f}\n"
                                    + (f"Confidence: {class_confidence:.1f}%\n" if class_confidence is not None else "")
                                    f"\nSensor Readings:\n"
                                    f"Accel (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}\n"
                                    f"Gyro (deg/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}\n"
                                    f"Temp: {temp:.1f}°C"
                                )
                            elif args.model_type == 'rf':
                                sms_msg = (
                                    f"WIND TURBINE ALERT\n"
                                    f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Type: {anomaly_name}\n"
                                    f"\nSensor Readings:\n"
                                    f"Accel (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}\n"
                                    f"Gyro (deg/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}\n"
                                    f"Temp: {temp:.1f}°C"
                                )
                            else:  # svm mode
                                sms_msg = (
                                    f"WIND TURBINE ALERT\n"
                                    f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Anomaly Score: {svm_score:.3f}\n"
                                    f"\nSensor Readings:\n"
                                    f"Accel (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}\n"
                                    f"Gyro (deg/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}\n"
                                    f"Temp: {temp:.1f}°C"
                                )
                            
                            send_sms_alert("+17782383531", sms_msg)
                    
                    print("\n" + "=" * 50)
                
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

class SensorBuffer:
    def __init__(self, window_size=1.0, expected_sample_rate=5):
        self.window_size = window_size
        self.expected_samples = int(window_size * expected_sample_rate)
        self.reset_buffer()
        
    def reset_buffer(self):
        self.timestamps = []
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        
    def add_reading(self, accel_x, accel_y, accel_z, 
                   gyro_x, gyro_y, gyro_z, timestamp):
        """Add a sensor reading to the buffer."""
        self.timestamps.append(timestamp)
        self.accel_x.append(accel_x)
        self.accel_y.append(accel_y)
        self.accel_z.append(accel_z)
        self.gyro_x.append(gyro_x)
        self.gyro_y.append(gyro_y)
        self.gyro_z.append(gyro_z)
        
        window_ready = len(self.timestamps) >= self.expected_samples
        if window_ready:
            self.process_window()
            self.reset_buffer()
        return window_ready

def load_model(model_path='model.pkl'):
    """Load the trained Random Forest model."""
    try:
        print(f"Loading model from {model_path}...")
        model_dict = joblib.load(model_path)
        return model_dict['model'], model_dict['features']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

class TurbineAnomalyDetector:
    def __init__(self, model_path='model.pkl'):
        self.model, self.features = load_model(model_path)
        if self.model is None:
            raise RuntimeError("Failed to load model")
        print("Model loaded successfully")
        
    def extract_features(self, buffer):
        """Extract all required features from the buffer."""
        # First get ACF features
        acf_features = {}
        for axis in ['x', 'y', 'z']:
            acf = buffer.compute_acf(f'accel_{axis}', nlags=4)
            for lag in range(1, 5):
                acf_features[f'accel_{axis}_acf_lag{lag}'] = acf[lag-1]
            
            acf = buffer.compute_acf(f'gyro_{axis}', nlags=4)
            for lag in range(1, 5):
                acf_features[f'gyro_{axis}_acf_lag{lag}'] = acf[lag-1]
        
        # Get statistical features
        stats = buffer.compute_statistics()
        
        # Combine all features in the correct order
        feature_values = []
        for feature in self.features:
            if feature in acf_features:
                feature_values.append(acf_features[feature])
            else:
                feature_values.append(stats[feature])
                
        return feature_values
        
    def predict(self, feature_values):
        """Make prediction using the loaded model."""
        try:
            prediction = self.model.predict([feature_values])[0]
            # Convert to binary anomaly detection
            # 0 = normal, 1/2 = anomaly types
            is_anomaly = prediction != 0
            return is_anomaly, prediction
        except Exception as e:
            print(f"Error making prediction: {e}")
            return True, -1  # Return as anomaly in case of error

if __name__ == "__main__":
    sys.exit(main())