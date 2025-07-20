"""
Intelligent response generation system that combines emotion analysis
with LLM capabilities to create empathetic, contextually appropriate responses.
"""

import logging
from typing import Dict, List, Optional
from .llm_response import ask_gemini

logger = logging.getLogger(__name__)

class EmpathyResponseGenerator:
    """Generates empathetic responses based on emotional context."""

    def __init__(self):
        self.system_prompts = self._load_system_prompts()
        self.response_templates = self._load_response_templates()

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts for different emotional contexts."""
        return {
            "default": """You are EmpathyAI, a compassionate mental health companion. 
                         Respond with warmth, understanding, and genuine care. 
                         Keep responses under 120 words. Be supportive but not clinical.
                         Use gentle, encouraging language. Acknowledge emotions: {emotion}.""",

            "sadness": """You are EmpathyAI, speaking to someone feeling sad or down.
                         Show deep empathy and validation. Offer gentle comfort and hope.
                         Remind them that sadness is temporary and they're not alone.
                         Keep under 120 words. Emotion context: {emotion}.""",

            "anger": """You are EmpathyAI, helping someone process anger or frustration.
                        Validate their feelings without encouraging harmful actions.
                        Help them find healthy ways to express and process anger.
                        Stay calm and grounding. Keep under 120 words. Emotion: {emotion}.""",

            "fear": """You are EmpathyAI, supporting someone experiencing fear or anxiety.
                       Offer reassurance and practical coping strategies.
                       Help them feel safe and grounded in the present moment.
                       Use calming, confident language. Under 120 words. Emotion: {emotion}.""",

            "joy": """You are EmpathyAI, celebrating positive emotions with someone.
                      Share in their happiness and help them savor the moment.
                      Encourage them to appreciate and remember this feeling.
                      Be warm and uplifting. Keep under 120 words. Emotion: {emotion}."""
        }

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load template responses for when LLM is unavailable."""
        return {
            "sadness": [
                "I can hear the sadness in your words, and I want you to know that it's okay to feel this way. Your emotions are valid, and you don't have to go through this alone. 💙",
                "It sounds like you're carrying some heavy feelings right now. Remember that sadness, while painful, is also a sign of your capacity to care deeply. I'm here with you. 🫂",
                "I see you're going through a difficult time. Please be gentle with yourself. These feelings won't last forever, even though they feel overwhelming right now. 🌸"
            ],
            "anger": [
                "I can sense your frustration, and those feelings are completely valid. It's human to feel angry sometimes. Let's take a moment to breathe together. 🌱",
                "Your anger is telling you something important about what matters to you. Let's explore what's underneath these feelings in a safe way. 💚",
                "I hear how upset you are. It takes courage to express difficult emotions. You're doing the right thing by talking about it instead of keeping it inside. 🤗"
            ],
            "fear": [
                "I can feel your worry, and I want you to know that you're safe here with me. Fear can be overwhelming, but you're braver than you think. 🦋",
                "Anxiety can make everything feel uncertain, but remember: you've faced difficult moments before and you're still here. That's strength. 🌟",
                "Your fears are real and valid. Let's take this one breath at a time. You don't have to face this alone. I'm right here with you. 🫧"
            ],
            "joy": [
                "I can feel your happiness radiating through your words! It's wonderful to witness your joy. Please take a moment to really savor this feeling. ✨",
                "Your positive energy is infectious! I'm so glad you're experiencing this happiness. You deserve these beautiful moments. 🌻",
                "What a delightful thing to hear! Your joy reminds me of all the good that exists in the world. Thank you for sharing this brightness. ☀️"
            ],
            "neutral": [
                "I'm here and I'm listening. Whatever you're going through, you don't have to face it alone. Your feelings matter. 🤗",
                "Thank you for trusting me with your thoughts. I'm honored to be part of your journey, whatever it may bring. 💜",
                "I hear you, and I'm with you in this moment. Sometimes just being acknowledged is enough. How are you really doing? 🌙"
            ]
        }

    def generate_response(self, user_text: str, fused_emotion: str, user_history: Optional[List] = None) -> Dict[str, str]:
        """
        Generate an empathetic response based on user input and emotional context.

        Args:
            user_text (str): The user's input message
            fused_emotion (str): The fused emotion-sentiment label
            user_history (List): Optional conversation history

        Returns:
            Dict: Contains the response and metadata
        """
        try:
            # Extract primary emotion from fused label
            primary_emotion = self._extract_primary_emotion(fused_emotion)

            # Build context-aware prompt
            prompt = self._build_prompt(user_text, fused_emotion, primary_emotion, user_history)

            # Generate response using LLM
            llm_response = ask_gemini(prompt, temperature=0.7)

            # Validate and enhance response if needed
            final_response = self._validate_and_enhance_response(
                llm_response, primary_emotion, user_text
            )

            return {
                "response": final_response,
                "emotion_detected": fused_emotion,
                "primary_emotion": primary_emotion,
                "generation_method": "llm" if "fallback" not in llm_response.lower() else "template",
                "confidence": self._calculate_response_confidence(final_response, fused_emotion)
            }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._fallback_response(user_text, fused_emotion)

    def _extract_primary_emotion(self, fused_emotion: str) -> str:
        """Extract primary emotion from fused label."""
        if not fused_emotion:
            return "neutral"

        # Handle fused emotions like "negative-sadness"
        parts = fused_emotion.lower().split("-")
        if len(parts) > 1:
            emotion = parts[-1]  # Take the emotion part
        else:
            emotion = parts[0]

        # Map to primary emotion categories
        emotion_mapping = {
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "anxiety": "fear",
            "joy": "joy",
            "happiness": "joy",
            "surprise": "neutral",
            "disgust": "anger",
            "love": "joy",
            "optimism": "joy",
            "pessimism": "sadness"
        }

        return emotion_mapping.get(emotion, "neutral")

    def _build_prompt(self, user_text: str, fused_emotion: str, primary_emotion: str, history: Optional[List]) -> str:
        """Build a context-aware prompt for the LLM."""
        # Select appropriate system prompt
        system_prompt = self.system_prompts.get(primary_emotion, self.system_prompts["default"])
        system_prompt = system_prompt.format(emotion=fused_emotion)

        # Add conversation history if available
        history_context = ""
        if history and len(history) > 0:
            recent_history = history[-3:]  # Last 3 interactions
            history_context = "\n\nRecent conversation context:\n"
            for i, item in enumerate(recent_history, 1):
                if isinstance(item, dict):
                    history_context += f"{i}. User: {item.get('user', '')}, AI: {item.get('ai', '')}\n"

        # Combine into final prompt
        full_prompt = f"""{system_prompt}

{history_context}

Current user message: "{user_text}"

Please respond with empathy, understanding, and genuine care. Focus on:
1. Acknowledging their emotional state ({fused_emotion})
2. Providing appropriate support and validation
3. Being warm but not overly clinical
4. Keeping response under 120 words

Response:"""

        return full_prompt

    def _validate_and_enhance_response(self, response: str, emotion: str, user_text: str) -> str:
        """Validate LLM response and enhance if needed."""
        if not response or len(response.strip()) < 10:
            return self._get_template_response(emotion, user_text)

        # Check for inappropriate content (basic filtering)
        inappropriate_phrases = [
            "i'm just an ai", "i'm not a therapist", "seek professional help immediately",
            "i can't help with that", "that's not my job"
        ]

        response_lower = response.lower()
        if any(phrase in response_lower for phrase in inappropriate_phrases):
            return self._get_template_response(emotion, user_text)

        # Ensure response isn't too long
        if len(response) > 200:
            sentences = response.split(". ")
            response = ". ".join(sentences[:2]) + "."

        return response.strip()

    def _get_template_response(self, emotion: str, user_text: str) -> str:
        """Get a template response for the given emotion."""
        import random
        templates = self.response_templates.get(emotion, self.response_templates["neutral"])
        return random.choice(templates)

    def _calculate_response_confidence(self, response: str, emotion: str) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.7

        # Increase confidence for longer, more detailed responses
        if len(response) > 50:
            base_confidence += 0.1

        # Increase confidence if emotion-specific words are present
        emotion_keywords = {
            "sadness": ["sad", "difficult", "understand", "support", "comfort"],
            "anger": ["frustrated", "angry", "valid", "breath", "calm"],
            "fear": ["anxiety", "worry", "safe", "brave", "strength"],
            "joy": ["happy", "wonderful", "celebrate", "joy", "bright"]
        }

        keywords = emotion_keywords.get(emotion, [])
        if keywords and any(keyword in response.lower() for keyword in keywords):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _fallback_response(self, user_text: str, emotion: str) -> Dict[str, str]:
        """Generate fallback response when all else fails."""
        primary_emotion = self._extract_primary_emotion(emotion)
        response = self._get_template_response(primary_emotion, user_text)

        return {
            "response": response,
            "emotion_detected": emotion,
            "primary_emotion": primary_emotion,
            "generation_method": "fallback_template",
            "confidence": 0.5
        }

# Global generator instance
_generator = None

def get_generator() -> EmpathyResponseGenerator:
    """Get or create global response generator."""
    global _generator
    if _generator is None:
        _generator = EmpathyResponseGenerator()
    return _generator

def craft_empathy_response(user_text: str, fused_emotion: str, history: Optional[List] = None) -> str:
    """
    Convenient function to generate empathetic response.

    Args:
        user_text (str): User's input
        fused_emotion (str): Detected emotion
        history (List): Conversation history

    Returns:
        str: Generated empathetic response
    """
    generator = get_generator()
    result = generator.generate_response(user_text, fused_emotion, history)
    return result["response"]
