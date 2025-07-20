"""
Advanced emotion detection using HuggingFace transformers.
Provides reliable emotion classification with confidence scoring.
"""

from transformers import pipeline
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """Singleton emotion detection class for efficient model loading."""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pipeline is None:
            try:
                self._pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1  # Use CPU for compatibility
                )
                logger.info("Emotion detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                raise

    def detect_emotion(self, text: str, min_length: int = 5) -> Dict:
        """
        Detect emotion in text with confidence scores.

        Args:
            text (str): Input text to analyze
            min_length (int): Minimum text length for analysis

        Returns:
            dict: Contains top emotion label and all scores
        """
        if not text or len(text.strip()) < min_length:
            return {
                "label": "neutral",
                "confidence": 0.5,
                "all_scores": [{"label": "neutral", "score": 0.5}]
            }

        try:
            # Truncate very long text to avoid memory issues
            text = text[:512] if len(text) > 512 else text

            results = self._pipeline(text, truncation=True)
            if not results:
                return self._default_result()

            # Sort by confidence score
            sorted_results = sorted(results[0], key=lambda x: x["score"], reverse=True)
            top_result = sorted_results[0]

            return {
                "label": top_result["label"],
                "confidence": round(top_result["score"], 3),
                "all_scores": [{"label": r["label"], "score": round(r["score"], 3)} for r in sorted_results]
            }

        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        """Return default emotion result for error cases."""
        return {
            "label": "neutral",
            "confidence": 0.5,
            "all_scores": [{"label": "neutral", "score": 0.5}]
        }

# Create global detector instance
_detector = None

def get_detector() -> EmotionDetector:
    """Get or create the global emotion detector instance."""
    global _detector
    if _detector is None:
        _detector = EmotionDetector()
    return _detector

def detect_emotion(text: str) -> Dict:
    """
    Convenient function to detect emotion in text.

    Args:
        text (str): Text to analyze

    Returns:
        dict: Emotion analysis results
    """
    detector = get_detector()
    return detector.detect_emotion(text)
