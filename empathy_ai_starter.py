
# EmpathyAI Starter Template - Your First Mental Health AI Assistant
# Perfect for beginners new to NLP and AI development

import streamlit as st
from transformers import pipeline
import cv2
import numpy as np
from datetime import datetime
import pandas as pd

# Initialize session state for conversation memory
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'mood_history' not in st.session_state:
    st.session_state.mood_history = []

# Load pre-trained models (cached for performance)
@st.cache_resource
def load_emotion_model():
    """Load emotion detection model from Hugging Face"""
    return pipeline("text-classification", 
                   model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_conversational_model():
    """Load conversational AI model"""
    return pipeline("text-generation", 
                   model="microsoft/DialoGPT-medium")

# Initialize models
emotion_classifier = load_emotion_model()
# conversation_model = load_conversational_model()  # Uncomment when ready

def analyze_emotion(text):
    """Analyze emotion from user text input"""
    try:
        result = emotion_classifier(text)
        emotion = result[0]['label']
        confidence = result[0]['score']
        return emotion, confidence
    except Exception as e:
        st.error(f"Error analyzing emotion: {e}")
        return "neutral", 0.0

def generate_empathetic_response(user_input, detected_emotion):
    """Generate contextual empathetic responses based on emotion"""

    responses = {
        "sadness": [
            "I hear that you're going through a difficult time. Your feelings are completely valid.",
            "It sounds like you're feeling really down right now. Would you like to talk about what's troubling you?",
            "I'm sorry you're feeling sad. Remember, it's okay to not be okay sometimes."
        ],
        "anger": [
            "I can sense you're feeling frustrated or angry. That's a natural human emotion.",
            "It sounds like something has really upset you. Would you like to share what's bothering you?",
            "Anger can be overwhelming. Let's try to work through this together."
        ],
        "fear": [
            "I understand you might be feeling anxious or scared. Those feelings are valid.",
            "It's natural to feel afraid sometimes. You're not alone in feeling this way.",
            "Fear can be paralyzing, but you're brave for reaching out."
        ],
        "joy": [
            "It's wonderful to hear you're feeling happy! What's brought you joy today?",
            "I'm so glad you're feeling positive. It's great to celebrate the good moments.",
            "Your happiness is contagious! Would you like to share what's making you feel so good?"
        ],
        "surprise": [
            "It sounds like something unexpected happened. How are you processing it?",
            "Surprises can be overwhelming. Would you like to talk through what happened?",
            "Life can be full of surprises. How are you feeling about this one?"
        ],
        "disgust": [
            "It sounds like something is really bothering you or making you uncomfortable.",
            "I can sense you're feeling put off by something. Want to talk about it?",
            "Sometimes things can really get under our skin. I'm here to listen."
        ],
        "neutral": [
            "Thank you for sharing with me. How are you feeling right now?",
            "I'm here to listen and support you. What's on your mind today?",
            "How has your day been? I'm here if you need someone to talk to."
        ]
    }

    import random
    emotion_responses = responses.get(detected_emotion.lower(), responses["neutral"])
    return random.choice(emotion_responses)

def log_mood(emotion, confidence):
    """Log user mood for tracking over time"""
    mood_entry = {
        'timestamp': datetime.now(),
        'emotion': emotion,
        'confidence': confidence
    }
    st.session_state.mood_history.append(mood_entry)

def display_mood_analytics():
    """Display basic mood analytics"""
    if not st.session_state.mood_history:
        st.info("Start chatting to see your mood analytics!")
        return

    df = pd.DataFrame(st.session_state.mood_history)

    # Basic stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Interactions", len(df))

    with col2:
        most_common_emotion = df['emotion'].mode().iloc[0] if not df.empty else "N/A"
        st.metric("Most Common Emotion", most_common_emotion)

    with col3:
        avg_confidence = df['confidence'].mean() if not df.empty else 0
        st.metric("Average Confidence", f"{avg_confidence:.2f}")

    # Simple emotion distribution
    emotion_counts = df['emotion'].value_counts()
    st.bar_chart(emotion_counts)

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="EmpathyAI - Your Mental Health Companion",
        page_icon="🤗",
        layout="wide"
    )

    st.title("🤗 EmpathyAI - Your Mental Health Companion")
    st.markdown("*An AI assistant that listens, understands, and cares*")

    # Sidebar with project info and controls
    with st.sidebar:
        st.header("About EmpathyAI")
        st.markdown("""
        EmpathyAI is a mental health support chatbot that:
        - 🧠 Analyzes your emotions in real-time
        - 💬 Provides empathetic responses
        - 📊 Tracks your mood over time
        - 🔒 Keeps your conversations private
        """)

        st.header("Mood Analytics")
        display_mood_analytics()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.mood_history = []
            st.rerun()

    # Main chat interface
    st.header("Chat with EmpathyAI")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "user" and "emotion" in message:
                st.caption(f"Detected emotion: {message['emotion']} (confidence: {message['confidence']:.2f})")

    # User input
    if prompt := st.chat_input("How are you feeling today? Share what's on your mind..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Analyze emotion
        emotion, confidence = analyze_emotion(prompt)

        # Add emotion data to the last user message
        st.session_state.messages[-1]["emotion"] = emotion
        st.session_state.messages[-1]["confidence"] = confidence

        # Log mood for analytics
        log_mood(emotion, confidence)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

        # Generate and display AI response
        response = generate_empathetic_response(prompt, emotion)

        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer with helpful resources
    st.markdown("---")
    st.markdown("### 🆘 Need Immediate Help?")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Crisis Hotlines:**")
        st.markdown("- 988 (Suicide Prevention)")
        st.markdown("- 1-800-366-8288 (Self-Injury)")

    with col2:
        st.markdown("**Online Resources:**")
        st.markdown("- [Mental Health America](https://mhanational.org)")
        st.markdown("- [NAMI](https://nami.org)")

    with col3:
        st.markdown("**Emergency:**")
        st.markdown("- 911 (US Emergency)")
        st.markdown("- Go to nearest ER")

if __name__ == "__main__":
    main()

# How to run this app:
# 1. Save this code as 'empathy_ai.py'
# 2. Install requirements: pip install streamlit transformers torch
# 3. Run: streamlit run empathy_ai.py
# 4. Open browser to localhost:8501

# Next steps for enhancement:
# 1. Add facial emotion recognition with OpenCV
# 2. Integrate with Gemini API for better conversations
# 3. Add n8n automation for daily check-ins
# 4. Implement conversation memory across sessions
# 5. Add mood visualization charts
# 6. Deploy to Streamlit Cloud for public access
