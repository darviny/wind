#!/usr/bin/env python3
"""
lcd_alert.py - LCD display module for sensor alert system.

This module provides functions to display alerts on a 1602 I2C LCD display.
The LCD has 2 lines of 16 characters each. Long messages are split across
lines or scrolled if necessary.

Typical usage:
    lcd = LCDAlert()
    lcd.display_alert("Anomaly Detected")
    
Configuration:
    - Default I2C address is 0x27
    - Uses default I2C port (1)
    - 16x2 character display
"""

import time
from RPLCD.i2c import CharLCD
from typing import Optional

class LCDAlert:
    """
    Handler for 1602 I2C LCD display alerts.
    
    Attributes:
        lcd: CharLCD instance for display control
        cols: Number of columns (default 16)
        rows: Number of rows (default 2)
    """
    
    def __init__(self, i2c_address: int = 0x27, port: int = 1, 
                 cols: int = 16, rows: int = 2):
        """
        Initialize the LCD display.
        
        Args:
            i2c_address: I2C address of LCD (default 0x27)
            port: I2C port number (default 1)
            cols: Number of columns (default 16)
            rows: Number of rows (default 2)
            
        Raises:
            RuntimeError: If LCD initialization fails
        """
        try:
            self.lcd = CharLCD(
                i2c_expander='PCF8574',
                address=i2c_address,
                port=port,
                cols=cols,
                rows=rows,
                dotsize=8
            )
            self.cols = cols
            self.rows = rows
            self.lcd.clear()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LCD: {e}")
    
    def display_alert(self, message: str, duration: Optional[float] = None):
        """
        Display an alert message on the LCD.
        
        Args:
            message: The message to display
            duration: How long to show message (in seconds, None for permanent)
            
        Returns:
            bool: True if successful, False if display failed
        """
        try:
            # Clear any existing message
            self.lcd.clear()
            
            # Split message into lines if needed
            if len(message) <= self.cols:
                # Short message - display on first line
                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(message)
            else:
                # Long message - split across lines
                line1 = message[:self.cols]
                line2 = message[self.cols:self.cols*2]
                
                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(line1)
                
                if line2:
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(line2[:self.cols])
            
            # If duration specified, clear after timeout
            if duration is not None:
                time.sleep(duration)
                self.lcd.clear()
            
            return True
            
        except Exception as e:
            print(f"Failed to display alert on LCD: {e}")
            return False
    
    def scroll_message(self, message: str, delay: float = 0.5):
        """
        Scroll a long message across the LCD.
        
        Args:
            message: The message to scroll
            delay: Delay between scroll steps in seconds
        """
        try:
            # Pad message for smooth scrolling
            padded_message = message + " " * self.cols
            
            for i in range(len(message) + 1):
                self.lcd.clear()
                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(padded_message[i:i+self.cols])
                time.sleep(delay)
                
            return True
            
        except Exception as e:
            print(f"Failed to scroll message on LCD: {e}")
            return False
    
    def clear(self):
        """Clear the LCD display."""
        try:
            self.lcd.clear()
            return True
        except Exception as e:
            print(f"Failed to clear LCD: {e}")
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.clear()
        except:
            pass


# Example usage
if __name__ == "__main__":
    try:
        # Initialize LCD
        lcd = LCDAlert()
        
        # Test basic alert
        lcd.display_alert("Test Alert!")
        time.sleep(2)
        
        # Test long message
        lcd.display_alert("This is a longer alert message that needs scrolling")
        time.sleep(2)
        
        # Test scrolling
        lcd.scroll_message("This message will scroll across the display")
        
    except Exception as e:
        print(f"Error in LCD test: {e}")