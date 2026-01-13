import smtplib
import ssl
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Ensure you have the twilio library installed: pip install twilio
try:
    from twilio.rest import Client
except ImportError:
    logging.warning("Twilio library not found. WhatsApp notifications will be disabled.")
    Client = None

# ==============================================================================
# --- NOTIFICATION ABSTRACTION LAYER ---
# ==============================================================================

class NotificationClient(ABC):
    """Abstract Base Class for Notification methods."""

    @abstractmethod
    def send_notification(self, subject: str, body: str) -> None:
        """Sends the notification."""
        pass

class EmailNotificationClient(NotificationClient):
    """Concrete implementation for sending Email notifications."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.sender_email = config['sender_email']
        self.sender_password = config['sender_password']
        self.recipient_email = config['recipient_email']
        logging.info("EmailNotificationClient initialized.")

    def send_notification(self, subject: str, body: str) -> None:
        """Sends an HTML email notification using MIME with improved SSL/TLS handling."""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg.attach(MIMEText(body, 'html'))

            context = ssl.create_default_context()

            # --- LOGIQUE DE CONNEXION CORRIGÉE ---
            if self.smtp_port == 465:
                # SSL Direct (Ancien standard, port 465)
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            else:
                # STARTTLS (Standard moderne, port 587 ou 25)
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context) # Passage en mode sécurisé
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            
            logging.info(f"Successfully sent EMAIL notification to {self.recipient_email}.")
        except Exception as e:
            logging.error(f"Failed to send EMAIL notification: {e}")
            
class WhatsAppNotificationClient(NotificationClient):
    """Concrete implementation for sending WhatsApp notifications via Twilio."""
    def __init__(self, config: Dict[str, Any]):
        if Client is None:
            raise RuntimeError("Twilio library is required for WhatsApp notifications.")
        self.config = config
        self.account_sid = config['account_sid']
        self.auth_token = config['auth_token']
        self.twilio_number = config['twilio_number']
        self.recipient_number = config['recipient_number']
        self.client = Client(self.account_sid, self.auth_token)
        logging.info("WhatsAppNotificationClient initialized.")

    def send_notification(self, subject: str, body: str) -> None:
        """Sends a WhatsApp message (Twilio only allows body, so subject is prepended)."""
        whatsapp_body = f"[{subject}]\n{body}"
        try:
            message = self.client.messages.create(
                from_=self.twilio_number,
                body=whatsapp_body,
                to=self.recipient_number
            )
            logging.info(f"Successfully sent WHATSAPP notification (SID: {message.sid}).")
        except Exception as e:
            logging.error(f"Failed to send WHATSAPP notification: {e}")

class NotificationManager:
    """Manages and routes notifications to multiple clients."""
    def __init__(self, config: Dict[str, Any]):
        self.clients = []
        if not config.get('enabled', False):
            logging.info("Notification system is disabled in config.")
            return

        for client_type in config.get('types', []):
            try:
                if client_type == "Email" and 'email' in config:
                    self.clients.append(EmailNotificationClient(config['email']))
                elif client_type == "WhatsApp" and 'whatsapp' in config:
                    self.clients.append(WhatsAppNotificationClient(config['whatsapp']))
                else:
                    logging.warning(f"Notification type '{client_type}' configured but missing required parameters.")
            except Exception as e:
                logging.error(f"Could not initialize {client_type} client: {e}")

    def notify(self, subject: str, body: str) -> None:
        """Sends the notification using all initialized clients."""
        if not self.clients:
            return

        logging.info(f"Sending notification to {len(self.clients)} clients.")
        for client in self.clients:
            client.send_notification(subject, body)
