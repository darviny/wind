#!/usr/bin/env python3
"""
sms_alert.py - SMS alerting module using Twilio API with cooldown period.

This module provides functionality to send SMS alerts when anomalies are detected,
with a cooldown period between alerts to prevent spam.

Environment variables required:
    - TWILIO_ACCOUNT_SID: Your Twilio account SID
    - TWILIO_AUTH_TOKEN: Your Twilio authentication token
    - TWILIO_FROM_PHONE: Your Twilio phone number to send from

Usage:
    from sms_alert import send_sms_alert
    
    # Send an alert
    success = send_sms_alert("+1234567890", "Anomaly detected!")
    
Configuration:
    Export your Twilio credentials as environment variables:
    export TWILIO_ACCOUNT_SID='your_account_sid'
    export TWILIO_AUTH_TOKEN='your_auth_token'
    export TWILIO_FROM_PHONE='+1234567890'
"""

import os
import time
import logging
from typing import Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track last alert time for cooldown
last_alert_time = 0
COOLDOWN_PERIOD = 10  # seconds

def get_twilio_client() -> Optional[Client]:
    """
    Initialize Twilio client using environment variables.
    
    Returns:
        Optional[Client]: Twilio client instance or None if credentials are missing
    """
    # Get credentials from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        logger.error("Missing Twilio credentials in environment variables")
        return None
    
    return Client(account_sid, auth_token)

def send_sms_alert(to_phone: str, message: str) -> bool:
    """
    Send an SMS alert using Twilio, respecting the cooldown period.
    
    Args:
        to_phone (str): Destination phone number in E.164 format (+1234567890)
        message (str): Alert message to send
    
    Returns:
        bool: True if SMS was sent successfully or skipped due to cooldown,
              False if there was an error
    """
    global last_alert_time
    
    # Check cooldown period
    current_time = time.time()
    time_since_last_alert = current_time - last_alert_time
    
    if time_since_last_alert < COOLDOWN_PERIOD:
        logger.info(f"Skipping alert - in cooldown period ({time_since_last_alert:.1f}s < {COOLDOWN_PERIOD}s)")
        return True  # Return True as this is expected behavior
    
    # Get the 'from' phone number from environment
    from_phone = os.getenv('TWILIO_FROM_PHONE')
    if not from_phone:
        logger.error("Missing TWILIO_FROM_PHONE in environment variables")
        return False
    
    # Initialize Twilio client
    client = get_twilio_client()
    if not client:
        return False
    
    try:
        # Send the message
        message = client.messages.create(
            body=message,
            from_=from_phone,
            to=to_phone
        )
        
        # Update last alert time
        last_alert_time = current_time
        
        logger.info(f"SMS alert sent successfully. SID: {message.sid}")
        return True
        
    except TwilioRestException as e:
        logger.error(f"Twilio API error: {e.msg}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending SMS: {str(e)}")
        return False

def set_cooldown_period(seconds: int):
    """
    Set the cooldown period between alerts.
    
    Args:
        seconds (int): Number of seconds to wait between alerts
    """
    global COOLDOWN_PERIOD
    COOLDOWN_PERIOD = seconds
    logger.info(f"Alert cooldown period set to {seconds} seconds")

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test the SMS alert functionality with cooldown
    try:
        # Check if required environment variables are set
        required_vars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_PHONE']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print("Error: Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\nPlease set them before running this script:")
            print("export TWILIO_ACCOUNT_SID='your_account_sid'")
            print("export TWILIO_AUTH_TOKEN='your_auth_token'")
            print("export TWILIO_FROM_PHONE='+1234567890'")
            sys.exit(1)
        
        # Test phone number (should be configured by user)
        test_phone = "+1234567890"  # Replace with actual phone number
        
        # Test multiple alerts with cooldown
        print(f"Testing alerts with {COOLDOWN_PERIOD}s cooldown...")
        
        # First alert
        print("\nSending first alert...")
        send_sms_alert(test_phone, "Test alert 1")
        
        # Second alert (should be blocked by cooldown)
        print("\nTrying to send second alert immediately...")
        send_sms_alert(test_phone, "Test alert 2")
        
        # Wait for cooldown
        print(f"\nWaiting {COOLDOWN_PERIOD} seconds...")
        time.sleep(COOLDOWN_PERIOD)
        
        # Third alert (should work)
        print("\nSending third alert after cooldown...")
        send_sms_alert(test_phone, "Test alert 3")
        
    except Exception as e:
        print(f"Error during testing: {e}") 