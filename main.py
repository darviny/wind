#!/usr/bin/env python3
"""
main.py - Main script for ADXL345 accelerometer monitoring with anomaly detection.

This script integrates all components of the monitoring system:
- Sensor reading from ADXL345 accelerometer
- Data buffering and feature calculation
- Anomaly detection
- Alerting

It can be run directly or integrated into a systemd service.
"""

import time
import signal
import sys
from datetime import datetime

# Import sensor reader
from sensor_reader import ADXL345Reader

# Import data handler
from data_handler import AccelerationBuffer, log_acceleration_to_csv

# Import anomaly detection
from anomaly_detector import DEFAULT_THRESHOLDS, check_anomaly

# Import alerting
from alert import send_alert, log_alert

# Global flag for clean shutdown
running = True

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global running
    print("\nShutting down monitoring system...")
    running = False

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def run_monitoring(sample_rate=20, window_size=1.0, use_svm=False):
    """
    Run the accelerometer monitoring system.
    
    Args:
        sample_rate (int): Samples per second (Hz). Default is 20.
        window_size (float): Window size in seconds for feature calculation. Default is 1.0.
        use_svm (bool): Use One-Class SVM for anomaly detection if True, 
                       threshold-based detection if False. Default is False.
    """
    global running
    
    print("Starting ADXL345 Accelerometer Monitoring System")
    print(f"Sample rate: {sample_rate} Hz, Window size: {window_size} seconds")
    print(f"Anomaly detection method: {'One-Class SVM' if use_svm else 'Threshold-based'}")
    
    # Initialize sensor
    sensor = None
    acceleration_buffer = None
    last_alert_time = time.time() - 60  # Initialize to avoid alert flood
    alert_cooldown = 10  # Seconds between alerts
    
    try:
        # Initialize the ADXL345 sensor
        print("Initializing ADXL345 sensor...")
        sensor = ADXL345Reader()
        print("Sensor initialized successfully")
        
        # Initialize the data buffer
        acceleration_buffer = AccelerationBuffer(
            window_size=window_size,
            expected_sample_rate=sample_rate
        )
        
        # Custom thresholds (optional)
        # acceleration_buffer.set_thresholds({
        #    "x_std": 0.05,
        #    "y_std": 0.05,
        #    "z_std": 0.05
        # })
        
        # Calculate sleep time between samples
        sleep_time = 1.0 / sample_rate
        
        print("\nMonitoring system active. Press Ctrl+C to stop.")
        print("-" * 50)
        
        # Main monitoring loop
        sample_count = 0
        start_time = time.time()
        
        while running:
            # Read acceleration data
            x, y, z = sensor.get_acceleration()
            timestamp = datetime.now()
            
            # Log raw acceleration data
            log_acceleration_to_csv(x, y, z, timestamp.isoformat())
            
            # Add data to buffer for feature calculation
            # Returns True if a window was processed
            window_processed = acceleration_buffer.add_reading(x, y, z, timestamp)
            
            # If a window was processed, check if it was anomalous
            if window_processed:
                # Get the last window's features
                features = acceleration_buffer._compute_features()
                
                # Check for anomalies
                is_anomaly, exceeded_features = check_anomaly(features, DEFAULT_THRESHOLDS)
                
                # Handle anomaly detection
                if is_anomaly:
                    current_time = time.time()
                    
                    # Only send alert if cooldown period has passed
                    if current_time - last_alert_time > alert_cooldown:
                        # Prepare alert message
                        subject = "Accelerometer Anomaly Detected"
                        message = f"Anomaly detected at {timestamp.isoformat()}\n\n"
                        message += "Exceeded thresholds:\n"
                        
                        for feature, (value, threshold) in exceeded_features.items():
                            message += f"- {feature}: {value:.4f} (threshold: {threshold})\n"
                        
                        message += f"\nCurrent readings: X={x:.4f}, Y={y:.4f}, Z={z:.4f}"
                        
                        # Send alert
                        send_alert(subject, message)
                        log_alert(subject, message)
                        
                        # Update last alert time
                        last_alert_time = current_time
            
            # Increment sample count
            sample_count += 1
            
            # Calculate actual sleep time to maintain precise sampling rate
            elapsed = time.time() - start_time
            ideal_elapsed = sample_count * sleep_time
            actual_sleep = max(0, ideal_elapsed - elapsed)
            
            # Sleep until next sample
            if actual_sleep > 0:
                time.sleep(actual_sleep)
        
        # End of monitoring loop
        print("\nMonitoring stopped.")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        # Clean up resources
        if acceleration_buffer:
            print("Processing remaining data in buffer...")
            acceleration_buffer.process_remaining_data()
        
        if sensor:
            print("Closing sensor connection...")
            sensor.close()
        
        print("Monitoring system shutdown complete.")
    
    return 0


if __name__ == "__main__":
    # When run directly, start the monitoring system
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ADXL345 Accelerometer Monitoring System")
    parser.add_argument("--rate", type=int, default=20, help="Sample rate in Hz (default: 20)")
    parser.add_argument("--window", type=float, default=1.0, help="Window size in seconds (default: 1.0)")
    parser.add_argument("--svm", action="store_true", help="Use One-Class SVM for anomaly detection")
    
    args = parser.parse_args()
    
    # Run the monitoring system
    sys.exit(run_monitoring(
        sample_rate=args.rate, 
        window_size=args.window, 
        use_svm=args.svm
    ))