"""
Google Gemini API integration for generating empathetic responses.
Handles API calls, rate limiting, and error recovery.
"""

import os
import logging
import time
from typing import Optional, Dict
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google-generativeai not installed. LLM responses will use fallback.")

logger = logging.getLogger(__name__)

class GeminiClient:
    """Handles Gemini API interactions with error handling and rate limiting."""

    def __init__(self):
        self.model = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini client with API key."""
        if not genai:
            logger.warning("Gemini client unavailable - using fallback responses")
            return

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Try streamlit secrets
                try:
                    import streamlit as st
                    api_key = st.secrets.get("GEMINI_API_KEY")
                except:
                    pass

            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("No Gemini API key found")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def generate_response(self, prompt: str, max_retries: int = 3, temperature: float = 0.7) -> str:
        """
        Generate response using Gemini API with retries and rate limiting.

        Args:
            prompt (str): The prompt to send to Gemini
            max_retries (int): Maximum retry attempts
            temperature (float): Response creativity (0.0-1.0)

        Returns:
            str: Generated response or fallback message
        """
        if not self.model:
            return self._fallback_response(prompt)

        # Rate limiting
        self._enforce_rate_limit()

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": 500,
                        "top_p": 0.9
                    }
                )

                if response and response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")

            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return self._fallback_response(prompt)

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _fallback_response(self, prompt: str) -> str:
        """Generate fallback response when Gemini is unavailable."""
        # Simple keyword-based fallback responses
        text_lower = prompt.lower()

        if any(word in text_lower for word in ["sad", "depressed", "down", "upset"]):
            return "I understand you're going through a tough time. It's okay to feel sad sometimes - these feelings are valid and temporary. Would you like to talk more about what's bothering you? 💙"

        elif any(word in text_lower for word in ["angry", "frustrated", "mad", "annoyed"]):
            return "It sounds like you're feeling frustrated right now. That's completely understandable. Take a deep breath with me. Sometimes talking through what's making us angry can help. I'm here to listen. 🫂"

        elif any(word in text_lower for word in ["anxious", "worried", "nervous", "stressed"]):
            return "I can sense you're feeling anxious. Anxiety can be overwhelming, but you're not alone in this. Try taking some slow, deep breaths. What's one thing that usually helps you feel calmer? 🌸"

        elif any(word in text_lower for word in ["happy", "excited", "joyful", "great", "wonderful"]):
            return "I'm so glad to hear you're feeling positive! It's wonderful when we experience joy. What's been the highlight of your day? I'd love to celebrate this moment with you! ✨"

        elif any(word in text_lower for word in ["tired", "exhausted", "drained", "overwhelmed"]):
            return "You sound really tired right now. It's important to acknowledge when we need rest. Have you been taking care of yourself lately? Sometimes we need to slow down and recharge. 🌙"

        else:
            return "Thank you for sharing with me. I'm here to listen and support you through whatever you're experiencing. Your feelings matter, and you're not alone. How can I help you today? 🤗"

# Global client instance
_client = None

def get_client() -> GeminiClient:
    """Get or create global Gemini client."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client

def ask_gemini(prompt: str, temperature: float = 0.7) -> str:
    """
    Convenient function to get Gemini response.

    Args:
        prompt (str): Prompt to send
        temperature (float): Response creativity

    Returns:
        str: Generated response
    """
    client = get_client()
    return client.generate_response(prompt, temperature=temperature)

def check_api_health() -> Dict[str, bool]:
    """Check if Gemini API is available and working."""
    client = get_client()

    if not client.model:
        return {"available": False, "reason": "Client not initialized"}

    try:
        test_response = client.generate_response("Hello", max_retries=1)
        is_working = len(test_response) > 0 and "fallback" not in test_response.lower()

        return {
            "available": is_working,
            "reason": "API working" if is_working else "API not responding properly"
        }
    except:
        return {"available": False, "reason": "API test failed"}
