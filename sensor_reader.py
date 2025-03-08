#!/usr/bin/env python3
"""
sensor_reader.py - A reusable module for interfacing with an MPU6050 accelerometer/gyroscope.

This module provides a class-based interface for reading acceleration, gyroscope,
and temperature data from an MPU6050 sensor connected to a Raspberry Pi via I2C.
"""

import board
from adafruit_mpu6050 import MPU6050
from typing import Tuple, Dict, Union

class MPU6050Reader:
    """
    A class to handle communication with an MPU6050 accelerometer/gyroscope.
    
    This class provides methods to initialize the sensor and read acceleration,
    gyroscope, and temperature data. It handles errors gracefully and is designed
    to be easily integrated into other projects.
    """
    
    def __init__(self, i2c_address=None):
        """
        Initialize communication with the MPU6050 sensor.
        
        Args:
            i2c_address: Optional I2C address if not using default.
                        Default is None (uses the default address 0x68).
        
        Raises:
            ImportError: If required libraries are not installed.
            RuntimeError: If sensor initialization fails.
        """
        try:
            # Initialize I2C interface
            self.i2c = board.I2C()  # Uses board.SCL and board.SDA
            
            # Initialize the MPU6050 sensor
            if i2c_address is not None:
                self.mpu = MPU6050(self.i2c, address=i2c_address)
            else:
                self.mpu = MPU6050(self.i2c)
                
        except ImportError as e:
            raise ImportError(
                f"Required library not found: {e}. "
                "Please install required libraries with: "
                "pip3 install adafruit-circuitpython-mpu6050"
            ) from e
            
        except Exception as e:
            raise RuntimeError(
                f"Error initializing MPU6050 sensor: {e}. "
                "Please check your wiring and I2C configuration."
            ) from e
    
    def get_acceleration(self, as_dict=False) -> Union[Tuple[float, float, float], Dict[str, float]]:
        """
        Get the current acceleration values from the sensor.
        
        Args:
            as_dict: If True, returns a dictionary with 'x', 'y', 'z' keys.
                    If False (default), returns a tuple (x, y, z).
        
        Returns:
            Either a tuple (x, y, z) or dictionary {'x': x, 'y': y, 'z': z}
            with acceleration values in m/s².
        """
        try:
            x, y, z = self.mpu.acceleration
            if as_dict:
                return {'x': x, 'y': y, 'z': z}
            return (x, y, z)
        except Exception as e:
            raise RuntimeError(f"Error reading acceleration data: {e}") from e

    def get_gyro(self, as_dict=False) -> Union[Tuple[float, float, float], Dict[str, float]]:
        """
        Get the current gyroscope values from the sensor.
        
        Args:
            as_dict: If True, returns a dictionary with 'x', 'y', 'z' keys.
                    If False (default), returns a tuple (x, y, z).
        
        Returns:
            Either a tuple (x, y, z) or dictionary {'x': x, 'y': y, 'z': z}
            with gyroscope values in rad/s.
        """
        try:
            x, y, z = self.mpu.gyro
            if as_dict:
                return {'x': x, 'y': y, 'z': z}
            return (x, y, z)
        except Exception as e:
            raise RuntimeError(f"Error reading gyroscope data: {e}") from e

    def get_temperature(self) -> float:
        """
        Get the current temperature reading from the sensor.
        
        Returns:
            Temperature in degrees Celsius.
        """
        try:
            return self.mpu.temperature
        except Exception as e:
            raise RuntimeError(f"Error reading temperature data: {e}") from e
    
    def close(self):
        """
        Clean up resources used by the sensor.
        """
        pass
    
    def __enter__(self):
        """Support for the context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for the context manager protocol."""
        self.close()


# Example usage (if this module is run directly)
if __name__ == "__main__":
    import time
    
    try:
        # Create an instance of the MPU6050Reader
        sensor = MPU6050Reader()
        print("MPU6050 sensor initialized successfully")
        
        print("Reading sensor data. Press Ctrl+C to exit.")
        
        # Read and print sensor data in a loop
        while True:
            # Get acceleration data
            ax, ay, az = sensor.get_acceleration()
            # Get gyroscope data
            gx, gy, gz = sensor.get_gyro()
            # Get temperature
            temp = sensor.get_temperature()
            
            print(f"Acceleration: X: {ax:.2f}, Y: {ay:.2f}, Z: {az:.2f} m/s²")
            print(f"Gyroscope:    X: {gx:.2f}, Y: {gy:.2f}, Z: {gz:.2f} rad/s")
            print(f"Temperature:  {temp:.1f}°C")
            print("------------------------")
            
            # Wait before reading again
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nExiting program...")
        
    except Exception as e:
        print(f"Error: {e}")