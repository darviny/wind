#!/usr/bin/env python3
"""
sms_alert.py - SMS alerting module using Twilio API.

This module provides functionality to send SMS alerts when anomalies are detected.
It requires the following environment variables to be set:
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
    Send an SMS alert using Twilio.
    
    Args:
        to_phone (str): Destination phone number in E.164 format (+1234567890)
        message (str): Alert message to send
    
    Returns:
        bool: True if SMS was sent successfully, False otherwise
    """
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
        
        logger.info(f"SMS alert sent successfully. SID: {message.sid}")
        return True
        
    except TwilioRestException as e:
        logger.error(f"Twilio API error: {e.msg}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending SMS: {str(e)}")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test the SMS alert functionality
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
        
        # Send test message
        print("Sending test SMS alert...")
        success = send_sms_alert(
            test_phone,
            "Test alert from MPU6050 monitoring system!"
        )
        
        if success:
            print("Test SMS sent successfully!")
        else:
            print("Failed to send test SMS")
            
    except Exception as e:
        print(f"Error during testing: {e}") 