"""
EmpathyAI - Modular Mental Health Assistant
A comprehensive AI system for empathetic conversation and emotional support.
"""

__version__ = "2.0.0"
__author__ = "Aditya Kumar Singh"

# Import main components for easy access
from .emotion import detect_emotion, get_detector
from .sentiment_fusion import fuse_sentiment_emotion, get_fusion  
from .llm_response import ask_gemini, get_client
from .response_generator import craft_empathy_response, get_generator
from .memory import create_memory_manager
from .n8n_integration import post_emotion_record, test_n8n_connection
from .auth import login, logout, require_authentication

__all__ = [
    'detect_emotion',
    'fuse_sentiment_emotion', 
    'ask_gemini',
    'craft_empathy_response',
    'create_memory_manager',
    'post_emotion_record',
    'login',
    'logout',
    'require_authentication'
]
