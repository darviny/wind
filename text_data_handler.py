#!/usr/bin/env python3
"""
test_data_handler.py - Test script for sensor_reader.py and data_handler.py.

This script continuously reads acceleration data from an ADXL345 sensor
and logs it to a CSV file at 5 Hz (every 0.2 seconds).
"""

import time
import signal
import sys
from datetime import datetime
from sensor_reader import ADXL345Reader
from data_handler import log_acceleration_to_csv

# Global flag for clean exit
running = True

# Set up clean exit handling
def signal_handler(sig, frame):
    global running
    print("\nStopping data collection...")
    running = False

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    sensor = None
    
    try:
        # Initialize the ADXL345 sensor
        print("Initializing ADXL345 sensor...")
        sensor = ADXL345Reader()
        print("Sensor initialized successfully")
        
        print("Starting data logging at 5 Hz. Press Ctrl+C to exit.")
        
        # Main loop for reading and logging data
        while running:
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Read acceleration data
            x, y, z = sensor.get_acceleration()
            
            # Log data to CSV
            if log_acceleration_to_csv(x, y, z, timestamp):
                print(f"Logged: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            else:
                print("Failed to log data")
            
            # Wait for next reading (5 Hz = 0.2 seconds)
            time.sleep(0.2)
        
        print("Data collection complete.")
    
    except KeyboardInterrupt:
        # Extra handling for Ctrl+C in case signal handler doesn't catch it
        print("\nData collection interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up if needed
        if sensor:
            sensor.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())