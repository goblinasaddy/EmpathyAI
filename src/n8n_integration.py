"""
n8n webhook integration for external analytics and workflow automation.
Sends emotion data to n8n workflows for dashboard creation and alerting.
"""

import os
import requests
import logging
from typing import Dict, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class N8nIntegration:
    """Handles n8n webhook integrations with retry logic."""

    def __init__(self):
        self.webhook_url = self._get_webhook_url()
        self.timeout = 5  # seconds
        self.max_retries = 2

    def _get_webhook_url(self) -> Optional[str]:
        """Get n8n webhook URL from environment or secrets."""
        webhook_url = os.getenv("N8N_WEBHOOK_URL")

        if not webhook_url:
            try:
                import streamlit as st
                webhook_url = st.secrets.get("N8N_WEBHOOK_URL")
            except:
                pass

        if webhook_url:
            logger.info("n8n webhook URL configured")
        else:
            logger.warning("No n8n webhook URL found - integration disabled")

        return webhook_url

    def send_emotion_data(self, user_id: str, emotion_data: Dict[str, Any]) -> bool:
        """
        Send emotion data to n8n workflow.

        Args:
            user_id (str): User identifier
            emotion_data (dict): Emotion analysis results

        Returns:
            bool: Success status
        """
        if not self.webhook_url:
            return False

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "event_type": "emotion_detected",
            "data": emotion_data
        }

        return self._send_webhook(payload)

    def send_conversation_data(self, user_id: str, conversation_data: Dict[str, Any]) -> bool:
        """
        Send conversation data to n8n workflow.

        Args:
            user_id (str): User identifier  
            conversation_data (dict): Conversation details

        Returns:
            bool: Success status
        """
        if not self.webhook_url:
            return False

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "event_type": "conversation_completed",
            "data": conversation_data
        }

        return self._send_webhook(payload)

    def send_user_analytics(self, user_id: str, analytics_data: Dict[str, Any]) -> bool:
        """
        Send user analytics to n8n workflow.

        Args:
            user_id (str): User identifier
            analytics_data (dict): Analytics summary

        Returns:
            bool: Success status  
        """
        if not self.webhook_url:
            return False

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "event_type": "user_analytics",
            "data": analytics_data
        }

        return self._send_webhook(payload)

    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Send webhook with retry logic.

        Args:
            payload (dict): Data to send

        Returns:
            bool: Success status
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "EmpathyAI/1.0"
                    }
                )

                if response.status_code in [200, 201, 202]:
                    logger.info(f"n8n webhook sent successfully (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(f"n8n webhook failed with status {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"n8n webhook timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"n8n webhook connection error (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"n8n webhook error (attempt {attempt + 1}): {e}")

            # Don't retry on the last attempt
            if attempt < self.max_retries - 1:
                import time
                time.sleep(1)  # Brief delay before retry

        logger.error("n8n webhook failed after all retries")
        return False

    def test_connection(self) -> Dict[str, Any]:
        """
        Test n8n webhook connection.

        Returns:
            dict: Connection test results
        """
        if not self.webhook_url:
            return {
                "connected": False,
                "error": "No webhook URL configured"
            }

        test_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "test_user",
            "event_type": "connection_test",
            "data": {"message": "EmpathyAI connection test"}
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=test_payload,
                timeout=self.timeout
            )

            return {
                "connected": response.status_code in [200, 201, 202],
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }

        except requests.exceptions.Timeout:
            return {"connected": False, "error": "Connection timeout"}
        except requests.exceptions.ConnectionError:
            return {"connected": False, "error": "Connection failed"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

# Global integration instance
_integration = None

def get_n8n_integration() -> N8nIntegration:
    """Get or create global n8n integration instance."""
    global _integration
    if _integration is None:
        _integration = N8nIntegration()
    return _integration

def post_emotion_record(user_id: str, emotion_label: str, confidence: float = None, 
                       message: str = None, session_id: str = None) -> bool:
    """
    Convenient function to post emotion record to n8n.

    Args:
        user_id (str): User identifier
        emotion_label (str): Detected emotion
        confidence (float): Confidence score
        message (str): User message
        session_id (str): Session identifier

    Returns:
        bool: Success status
    """
    integration = get_n8n_integration()

    emotion_data = {
        "emotion": emotion_label,
        "confidence": confidence or 0.5,
        "message": message or "",
        "session_id": session_id or ""
    }

    return integration.send_emotion_data(user_id, emotion_data)

def post_conversation_summary(user_id: str, session_id: str, 
                            message_count: int, emotions: list, 
                            duration_minutes: float = None) -> bool:
    """
    Post conversation summary to n8n.

    Args:
        user_id (str): User identifier
        session_id (str): Session identifier
        message_count (int): Number of messages
        emotions (list): List of emotions detected
        duration_minutes (float): Conversation duration

    Returns:
        bool: Success status
    """
    integration = get_n8n_integration()

    conversation_data = {
        "session_id": session_id,
        "message_count": message_count,
        "emotions_detected": emotions,
        "duration_minutes": duration_minutes,
        "summary": f"Conversation with {message_count} messages, emotions: {', '.join(emotions)}"
    }

    return integration.send_conversation_data(user_id, conversation_data)

def test_n8n_connection() -> Dict[str, Any]:
    """Test n8n webhook connection."""
    integration = get_n8n_integration()
    return integration.test_connection()
