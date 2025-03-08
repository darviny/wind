#!/usr/bin/env python3
"""
test_sensor.py - Tests an MPU6050 accelerometer/gyroscope connected to a Raspberry Pi via I2C.
"""

import time
import signal
import sys

# Try to import the necessary libraries
try:
    import board
    from adafruit_mpu6050 import MPU6050
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install required libraries with:")
    print("pip3 install adafruit-circuitpython-mpu6050")
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
        # Initialize the MPU6050 sensor
        mpu = MPU6050(i2c)
        print("MPU6050 sensor initialized successfully")
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        print("Please check your wiring and I2C configuration")
        sys.exit(1)
    
    print("Reading accelerometer and gyroscope data. Press Ctrl+C to exit.")
    
    # Main loop to read and print sensor data
    while True:
        try:
            # Read the acceleration and gyroscope values
            accel = mpu.acceleration
            gyro = mpu.gyro
            temp = mpu.temperature
            
            # Print the values
            print(f"Acceleration: X: {accel[0]:.2f}, Y: {accel[1]:.2f}, Z: {accel[2]:.2f} m/s²")
            print(f"Gyroscope:    X: {gyro[0]:.2f}, Y: {gyro[1]:.2f}, Z: {gyro[2]:.2f} rad/s")
            print(f"Temperature:  {temp:.2f}°C")
            print("------------------------")
            
            # Wait for 0.5 seconds
            time.sleep(0.5)
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            time.sleep(0.5)  # Wait before trying again

if __name__ == "__main__":
    main()