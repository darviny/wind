#!/usr/bin/env python3
"""
alert.py - Module for sending alerts from sensor anomaly detection systems.

This module provides a simple interface for sending alerts when anomalies
are detected. Currently outputs alerts to the console, but is designed to be
easily extended to email, SMS, or other notification channels.
"""

import time
from datetime import datetime


def send_alert(subject, message, alert_type="console"):
    """
    Send an alert with the specified subject and message.
    
    Args:
        subject (str): Alert subject line.
        message (str): Alert message content.
        alert_type (str): Type of alert to send. Currently supports:
                         - "console" (default): Print to console
    
    Returns:
        bool: True if alert was sent successfully, False otherwise.
    
    Example:
        >>> from alert import send_alert
        >>> send_alert("High Vibration Detected", "X-axis vibration exceeded threshold")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if alert_type == "console":
        # Print alert to console
        print("\n" + "!" * 60)
        print(f"ALERT [{timestamp}]: {subject}")
        print("-" * 60)
        print(message)
        print("!" * 60 + "\n")
        return True
        
    elif alert_type == "email":
        # Placeholder for email functionality
        # This section would be implemented when email alerting is needed
        """
        # Example implementation (to be completed later):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        sender_email = "alerts@example.com"
        receiver_email = "user@example.com"
        password = "your_password"  # Better to load from environment or config
        
        msg = MIMEMultipart()
        msg["Subject"] = f"SENSOR ALERT: {subject}"
        msg["From"] = sender_email
        msg["To"] = receiver_email
        
        body = f"Alert Time: {timestamp}\n\n{message}"
        msg.attach(MIMEText(body, "plain"))
        
        try:
            server = smtplib.SMTP("smtp.example.com", 587)
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
        """
        print("Email alerts not yet implemented")
        return False
        
    elif alert_type == "sms":
        # Placeholder for SMS functionality
        """
        # Example implementation with Twilio (to be completed later):
        from twilio.rest import Client
        
        account_sid = "your_account_sid"
        auth_token = "your_auth_token"
        from_number = "+1234567890"
        to_number = "+1987654321"
        
        try:
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=f"ALERT: {subject}\n{message}",
                from_=from_number,
                to=to_number
            )
            return True
        except Exception as e:
            print(f"Failed to send SMS: {e}")
            return False
        """
        print("SMS alerts not yet implemented")
        return False
    
    else:
        print(f"Unsupported alert type: {alert_type}")
        return False


def log_alert(subject, message, filename="alerts.log"):
    """
    Log an alert to a file.
    
    Args:
        subject (str): Alert subject.
        message (str): Alert message.
        filename (str): Path to log file. Default is "alerts.log".
    
    Returns:
        bool: True if logged successfully, False otherwise.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(filename, "a") as f:
            f.write(f"[{timestamp}] {subject}\n")
            f.write(f"{message}\n")
            f.write("-" * 40 + "\n")
        return True
    except Exception as e:
        print(f"Failed to log alert: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Example alert for testing
    test_subject = "Test Alert"
    test_message = "This is a test alert message to demonstrate functionality."
    
    # Send alert to console
    send_alert(test_subject, test_message)
    
    # Log alert to file
    log_alert(test_subject, test_message)
    
    print("Alert module test complete.")