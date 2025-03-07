#!/usr/bin/env python3
"""
sensor_reader.py - A reusable module for interfacing with an ADXL345 accelerometer.

This module provides a class-based interface for reading acceleration data
from an ADXL345 accelerometer connected to a Raspberry Pi via I2C.
"""

import board
import adafruit_adxl34x
from typing import Tuple, Dict, Union


class ADXL345Reader:
    """
    A class to handle communication with an ADXL345 accelerometer.
    
    This class provides methods to initialize the sensor and read acceleration data.
    It handles errors gracefully and is designed to be easily integrated into other projects.
    """
    
    def __init__(self, i2c_address=None):
        """
        Initialize communication with the ADXL345 accelerometer.
        
        Args:
            i2c_address: Optional I2C address if not using default.
                         Default is None (uses the default address 0x53).
        
        Raises:
            ImportError: If required libraries are not installed.
            RuntimeError: If sensor initialization fails.
        """
        try:
            # Initialize I2C interface
            self.i2c = board.I2C()  # Uses board.SCL and board.SDA
            
            # Initialize the ADXL345 sensor
            if i2c_address is not None:
                self.accelerometer = adafruit_adxl34x.ADXL345(self.i2c, address=i2c_address)
            else:
                self.accelerometer = adafruit_adxl34x.ADXL345(self.i2c)
                
            # Configure the sensor (optional settings)
            # Enable all axes
            self.accelerometer.enable_x = True
            self.accelerometer.enable_y = True
            self.accelerometer.enable_z = True
            
            # Set data rate (optional, default is usually fine)
            # self.accelerometer.data_rate = adafruit_adxl34x.DataRate.RATE_100_HZ
            
        except ImportError as e:
            raise ImportError(
                f"Required library not found: {e}. "
                "Please install required libraries with: "
                "pip3 install adafruit-circuitpython-adxl34x"
            ) from e
            
        except Exception as e:
            raise RuntimeError(
                f"Error initializing ADXL345 sensor: {e}. "
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
            
        Raises:
            RuntimeError: If there's an error reading from the sensor.
        """
        try:
            # Read acceleration data
            x, y, z = self.accelerometer.acceleration
            
            # Return the data in the requested format
            if as_dict:
                return {'x': x, 'y': y, 'z': z}
            else:
                return (x, y, z)
                
        except Exception as e:
            raise RuntimeError(f"Error reading acceleration data: {e}") from e
    
    def close(self):
        """
        Clean up resources used by the sensor.
        
        This method should be called when done using the sensor,
        especially in a context manager scenario.
        """
        # The adafruit library doesn't require explicit cleanup,
        # but we include this method for future compatibility
        # and to follow good resource management practices
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
        # Create an instance of the ADXL345Reader
        sensor = ADXL345Reader()
        print("ADXL345 sensor initialized successfully")
        
        print("Reading accelerometer data. Press Ctrl+C to exit.")
        
        # Read and print sensor data in a loop
        while True:
            # Get acceleration data as a tuple
            x, y, z = sensor.get_acceleration()
            print(f"X: {x:.2f} m/s²  Y: {y:.2f} m/s²  Z: {z:.2f} m/s²")
            
            # Alternatively, get data as a dictionary
            # data = sensor.get_acceleration(as_dict=True)
            # print(f"X: {data['x']:.2f} m/s²  Y: {data['y']:.2f} m/s²  Z: {data['z']:.2f} m/s²")
            
            # Wait before reading again
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nExiting program...")
        
    except Exception as e:
        print(f"Error: {e}")