#!/usr/bin/env python3
"""
test_sensor.py - Tests an ADXL345 accelerometer connected to a Raspberry Pi via I2C.
"""

import time
import signal
import sys

# Try to import the necessary libraries
try:
    import board
    import adafruit_adxl34x
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install required libraries with:")
    print("pip3 install adafruit-circuitpython-adxl34x")
    sys.exit(1)

# Set up clean exit handling
def signal_handler(sig, frame):
    print("\nExiting program...")
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    # Initialize the I2C interface
    try:
        i2c = board.I2C()  # Uses board.SCL and board.SDA
        # Initialize the ADXL345 sensor
        accelerometer = adafruit_adxl34x.ADXL345(i2c)
        print("ADXL345 sensor initialized successfully")
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        print("Please check your wiring and I2C configuration")
        sys.exit(1)
    
    print("Reading accelerometer data. Press Ctrl+C to exit.")
    
    # Main loop to read and print sensor data
    while True:
        try:
            # Read the acceleration values
            x, y, z = accelerometer.acceleration
            # Print the values
            print(f"X: {x:.2f} m/s²  Y: {y:.2f} m/s²  Z: {z:.2f} m/s²")
            # Wait for 0.5 seconds
            time.sleep(0.5)
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            time.sleep(0.5)  # Wait before trying again

if __name__ == "__main__":
    main()