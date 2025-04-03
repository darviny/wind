#!/usr/bin/env python3
import os
import time
from twilio.rest import Client

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_PHONE = os.getenv('TWILIO_FROM_PHONE')

COOLDOWN_PERIOD = 10  # seconds
last_alert_time = None

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms_alert(to_phone, message):
    global last_alert_time, client
    
    if last_alert_time is not None:
        time_since_last_alert = time.time() - last_alert_time
        if time_since_last_alert < COOLDOWN_PERIOD:
            print(f"Skipping alert - in cooldown period ({time_since_last_alert:.1f}s < {COOLDOWN_PERIOD}s)")
            return False
    
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_FROM_PHONE,
            to=to_phone
        )
        print(f"SMS alert sent successfully. SID: {message.sid}")
        last_alert_time = time.time()
        return True
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False

def set_cooldown_period(seconds):
    global COOLDOWN_PERIOD
    COOLDOWN_PERIOD = seconds