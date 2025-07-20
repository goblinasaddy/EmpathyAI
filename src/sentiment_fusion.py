"""
Multi-model sentiment fusion for nuanced emotion detection.
Combines base sentiment with emotional context for richer understanding.
"""

from transformers import pipeline
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SentimentFusion:
    """Combines multiple sentiment models for enhanced accuracy."""

    def __init__(self):
        self.base_model = None
        self.nuanced_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize both sentiment analysis models."""
        try:
            # Base sentiment model
            self.base_model = pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english",
                device=-1
            )
            logger.info("Base sentiment model loaded")

            # Nuanced tone model
            self.nuanced_model = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            logger.info("Nuanced sentiment model loaded")

        except Exception as e:
            logger.error(f"Failed to initialize sentiment models: {e}")
            # Fallback to single model
            try:
                self.base_model = pipeline("sentiment-analysis", device=-1)
                logger.info("Loaded fallback sentiment model")
            except:
                logger.error("All sentiment models failed to load")

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Perform multi-model sentiment analysis.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Combined sentiment analysis
        """
        if not text or len(text.strip()) < 3:
            return self._default_sentiment()

        try:
            # Truncate long text
            text = text[:512] if len(text) > 512 else text

            base_result = self._get_base_sentiment(text)
            nuanced_result = self._get_nuanced_sentiment(text)

            return {
                "base_sentiment": base_result,
                "nuanced_sentiment": nuanced_result,
                "combined_label": self._combine_sentiments(base_result, nuanced_result)
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._default_sentiment()

    def _get_base_sentiment(self, text: str) -> Dict:
        """Get base sentiment analysis."""
        if self.base_model:
            try:
                result = self.base_model(text, truncation=True)[0]
                return {
                    "label": result["label"].lower(),
                    "confidence": round(result["score"], 3)
                }
            except:
                pass
        return {"label": "neutral", "confidence": 0.5}

    def _get_nuanced_sentiment(self, text: str) -> Dict:
        """Get nuanced sentiment analysis."""
        if self.nuanced_model:
            try:
                result = self.nuanced_model(text, truncation=True)[0]
                # Map Cardiff model labels to standard format
                label_mapping = {
                    "LABEL_0": "negative",
                    "LABEL_1": "neutral", 
                    "LABEL_2": "positive"
                }
                mapped_label = label_mapping.get(result["label"], result["label"].lower())

                return {
                    "label": mapped_label,
                    "confidence": round(result["score"], 3)
                }
            except:
                pass
        return {"label": "neutral", "confidence": 0.5}

    def _combine_sentiments(self, base: Dict, nuanced: Dict) -> str:
        """Combine base and nuanced sentiment into single label."""
        base_label = base.get("label", "neutral")
        nuanced_label = nuanced.get("label", "neutral")

        # Weight by confidence
        base_conf = base.get("confidence", 0.5)
        nuanced_conf = nuanced.get("confidence", 0.5)

        if base_conf > 0.8:
            return base_label
        elif nuanced_conf > 0.8:
            return nuanced_label
        elif base_label == nuanced_label:
            return base_label
        else:
            # Take the more confident result
            return base_label if base_conf >= nuanced_conf else nuanced_label

    def _default_sentiment(self) -> Dict:
        """Default sentiment for error cases."""
        return {
            "base_sentiment": {"label": "neutral", "confidence": 0.5},
            "nuanced_sentiment": {"label": "neutral", "confidence": 0.5},
            "combined_label": "neutral"
        }

# Global fusion instance
_fusion = None

def get_fusion() -> SentimentFusion:
    """Get or create global sentiment fusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = SentimentFusion()
    return _fusion

def fuse_sentiment_emotion(text: str, emotion_label: str = None) -> str:
    """
    Create fused emotion-sentiment label.

    Args:
        text (str): Input text
        emotion_label (str): Emotion from emotion detection

    Returns:
        str: Fused label like 'negative-sadness' or 'positive-joy'
    """
    try:
        fusion = get_fusion()
        sentiment_result = fusion.analyze_sentiment(text)
        base_sentiment = sentiment_result["combined_label"]

        if emotion_label:
            return f"{base_sentiment}-{emotion_label}"
        else:
            return base_sentiment

    except Exception as e:
        logger.error(f"Sentiment fusion failed: {e}")
        return f"neutral-{emotion_label}" if emotion_label else "neutral"
