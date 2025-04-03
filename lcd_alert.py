#!/usr/bin/env python3

from RPLCD.i2c import CharLCD

class LCDAlert:
    def __init__(self, i2c_address=0x27, port=1, cols=16, rows=2):
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
    
    def display_alert(self, message):
        self.lcd.clear()
            
        if len(message) <= self.cols:
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(message)
        else:
            line1 = message[:self.cols]
            line2 = message[self.cols:self.cols*2]
            
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1)
            
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:self.cols])