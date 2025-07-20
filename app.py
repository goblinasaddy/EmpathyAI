"""
EmpathyAI 2.0 - Professional Mental Health Assistant
A modular, production-ready AI system for empathetic conversation.
"""

import streamlit as st
import logging
import uuid
from datetime import datetime
from typing import Dict, List

# Import our modular components
from src.auth import require_authentication, logout, get_current_user_id
from src.emotion import detect_emotion
from src.sentiment_fusion import fuse_sentiment_emotion
from src.response_generator import craft_empathy_response, get_generator
from src.memory import create_memory_manager
from src.n8n_integration import post_emotion_record, test_n8n_connection
from src.llm_response import check_api_health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="EmpathyAI - Your Mental Health Companion",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4A90E2;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .ai-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class EmpathyAIApp:
    """Main application class for EmpathyAI."""

    def __init__(self):
        self.user_info = None
        self.user_id = None
        self.memory = None
        self.session_id = None
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session state variables."""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())

        self.session_id = st.session_state.current_session_id

        if 'message_count' not in st.session_state:
            st.session_state.message_count = 0

        if 'emotions_detected' not in st.session_state:
            st.session_state.emotions_detected = []

    def authenticate_user(self):
        """Handle user authentication."""
        self.user_info = require_authentication(
            "Welcome to EmpathyAI 💙 Please log in to start your journey toward better mental wellness."
        )
        self.user_id = get_current_user_id()

        if self.user_id:
            self.memory = create_memory_manager(self.user_id)

    def render_sidebar(self):
        """Render the application sidebar."""
        with st.sidebar:
            st.markdown("### 💙 EmpathyAI Dashboard")

            # User info
            st.markdown(f"**Welcome, {self.user_info.get('name', 'Friend')}!** 👋")

            # Session info
            st.markdown("---")
            st.markdown("### 📊 Current Session")
            st.metric("Messages", st.session_state.message_count)
            st.metric("Session Time", self._get_session_duration())

            # Recent emotions
            if st.session_state.emotions_detected:
                st.markdown("### 🎭 Emotions Detected")
                for emotion in st.session_state.emotions_detected[-5:]:
                    confidence = emotion.get('confidence', 0.5)
                    color = self._get_emotion_color(emotion['label'])
                    st.markdown(
                        f'<span class="emotion-badge" style="background-color: {color}; color: white;">'
                        f'{emotion["label"]} ({confidence:.1%})</span>',
                        unsafe_allow_html=True
                    )

            # System status
            st.markdown("---")
            st.markdown("### ⚙️ System Status")
            self._show_system_status()

            # Analytics
            if self.memory:
                self._show_user_analytics()

            # Logout
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()

    def _get_session_duration(self) -> str:
        """Get formatted session duration."""
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now()

        duration = datetime.now() - st.session_state.session_start_time
        minutes = int(duration.total_seconds() // 60)
        return f"{minutes} min"

    def _get_emotion_color(self, emotion: str) -> str:
        """Get color for emotion badge."""
        color_map = {
            'joy': '#4CAF50',
            'sadness': '#2196F3', 
            'anger': '#F44336',
            'fear': '#FF9800',
            'surprise': '#9C27B0',
            'disgust': '#795548',
            'neutral': '#9E9E9E'
        }
        return color_map.get(emotion.lower(), '#9E9E9E')

    def _show_system_status(self):
        """Show system component status."""
        # Check LLM API
        api_health = check_api_health()
        api_status = "✅" if api_health.get("available") else "⚠️"
        st.text(f"{api_status} LLM API")

        # Check n8n integration
        n8n_status = test_n8n_connection()
        n8n_icon = "✅" if n8n_status.get("connected") else "⚠️"
        st.text(f"{n8n_icon} n8n Integration")

        # Memory system
        memory_icon = "✅" if self.memory else "❌"
        st.text(f"{memory_icon} Memory System")

    def _show_user_analytics(self):
        """Show user analytics in sidebar."""
        try:
            patterns = self.memory.get_emotion_patterns(days=7)
            if patterns.get("total_entries", 0) > 0:
                st.markdown("### 📈 7-Day Insights")
                st.metric("Total Conversations", patterns["total_entries"])

                # Top emotion
                top_emotions = sorted(
                    patterns["patterns"].items(),
                    key=lambda x: x[1]["frequency"],
                    reverse=True
                )
                if top_emotions:
                    top_emotion, stats = top_emotions[0]
                    st.metric("Most Common Emotion", top_emotion, f"{stats['percentage']}%")
        except Exception as e:
            logger.error(f"Analytics error: {e}")

    def render_main_chat(self):
        """Render the main chat interface."""
        # Header
        st.markdown('<h1 class="main-header">💙 EmpathyAI - Your Mental Health Companion</h1>', 
                   unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p>I'm here to listen, understand, and support you through whatever you're experiencing. 
            Your feelings matter, and you're not alone. 🤗</p>
        </div>
        """, unsafe_allow_html=True)

        # Display conversation history
        self._display_conversation_history()

        # Chat input
        self._handle_chat_input()

    def _display_conversation_history(self):
        """Display the conversation history."""
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                with st.container():
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
            else:
                with st.container():
                    st.markdown(
                        f'<div class="chat-message ai-message">'
                        f'<strong>EmpathyAI:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )

                    # Show emotion metadata if available
                    if "emotion_data" in message:
                        emotion_data = message["emotion_data"]
                        st.caption(
                            f"🎭 Detected: {emotion_data.get('emotion_detected', 'unknown')} "
                            f"(Confidence: {emotion_data.get('confidence', 0):.1%})"
                        )

    def _handle_chat_input(self):
        """Handle chat input and generate responses."""
        user_input = st.chat_input("Share what's on your mind... 💭")

        if user_input:
            self._process_user_message(user_input)

    def _process_user_message(self, user_input: str):
        """Process user message and generate AI response."""
        try:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })

            # Detect emotion
            emotion_result = detect_emotion(user_input)
            emotion_label = emotion_result.get("label", "neutral")
            confidence = emotion_result.get("confidence", 0.5)

            # Fuse sentiment with emotion
            fused_emotion = fuse_sentiment_emotion(user_input, emotion_label)

            # Get conversation history for context
            history = []
            if self.memory:
                history = self.memory.get_conversation_history(self.session_id, limit=3)

            # Generate empathetic response
            ai_response = craft_empathy_response(user_input, fused_emotion, history)

            # Store emotion data
            emotion_data = {
                "emotion_detected": fused_emotion,
                "confidence": confidence,
                "primary_emotion": emotion_label
            }

            # Add AI response to history
            st.session_state.conversation_history.append({
                "role": "assistant", 
                "content": ai_response,
                "timestamp": datetime.now().isoformat(),
                "emotion_data": emotion_data
            })

            # Update session tracking
            st.session_state.message_count += 1
            st.session_state.emotions_detected.append({
                "label": emotion_label,
                "confidence": confidence,
                "fused": fused_emotion
            })

            # Save to memory
            if self.memory:
                self.memory.add_emotion_record(
                    emotion_label=fused_emotion,
                    confidence=confidence,
                    message=user_input,
                    response=ai_response,
                    session_id=self.session_id
                )

                self.memory.add_conversation_context(
                    user_message=user_input,
                    ai_response=ai_response,
                    session_id=self.session_id
                )

            # Send to n8n for analytics
            post_emotion_record(
                user_id=self.user_id,
                emotion_label=fused_emotion,
                confidence=confidence,
                message=user_input,
                session_id=self.session_id
            )

            # Rerun to show new messages
            st.rerun()

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error("I'm having trouble processing your message right now. Please try again.")

    def run(self):
        """Run the main application."""
        try:
            self.authenticate_user()
            self.render_sidebar()
            self.render_main_chat()

        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("Something went wrong. Please refresh the page and try again.")

# Run the application
if __name__ == "__main__":
    app = EmpathyAIApp()
    app.run()
