"""
Authentication system using Google OAuth with streamlit-google-auth.
Provides secure user authentication with session management.
"""

import os
import logging
from typing import Dict, Tuple, Optional
import streamlit as st

logger = logging.getLogger(__name__)

# Optional Google Auth support
try:
    from streamlit_google_auth import Authenticate
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    logger.warning("streamlit-google-auth not available - using simple auth fallback")

class AuthManager:
    """Manages user authentication with multiple backends."""

    def __init__(self):
        self.authenticator = None
        self.auth_method = "simple"  # Default fallback
        self._init_auth_system()

    def _init_auth_system(self):
        """Initialize the authentication system."""
        if GOOGLE_AUTH_AVAILABLE and self._google_auth_configured():
            self._init_google_auth()
        else:
            logger.info("Using simple authentication fallback")
            self.auth_method = "simple"

    def _google_auth_configured(self) -> bool:
        """Check if Google OAuth is properly configured."""
        try:
            # Check for credentials file
            creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
            if os.path.exists(creds_path):
                return True

            # Check streamlit secrets
            if hasattr(st, 'secrets'):
                required_secrets = ["google_client_id", "google_client_secret", "cookie_key"]
                return all(secret in st.secrets for secret in required_secrets)

            return False
        except Exception as e:
            logger.error(f"Error checking Google auth config: {e}")
            return False

    def _init_google_auth(self):
        """Initialize Google OAuth authentication."""
        try:
            # Try file-based credentials first
            creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")

            if os.path.exists(creds_path):
                self.authenticator = Authenticate(
                    secret_credentials_path=creds_path,
                    cookie_name="empathy_auth",
                    cookie_key=st.secrets.get("cookie_key", "empathy_secret_key"),
                    redirect_uri=st.secrets.get("redirect_uri", "http://localhost:8501")
                )
            else:
                # Use streamlit secrets
                # Create temporary credentials file
                import json
                import tempfile

                creds_data = {
                    "web": {
                        "client_id": st.secrets["google_client_id"],
                        "client_secret": st.secrets["google_client_secret"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                }

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    json.dump(creds_data, f)
                    temp_creds_path = f.name

                self.authenticator = Authenticate(
                    secret_credentials_path=temp_creds_path,
                    cookie_name="empathy_auth",
                    cookie_key=st.secrets["cookie_key"],
                    redirect_uri=st.secrets.get("redirect_uri", "http://localhost:8501")
                )

            self.auth_method = "google"
            logger.info("Google OAuth authentication initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Google auth: {e}")
            self.auth_method = "simple"

    def login(self) -> Tuple[bool, Dict]:
        """
        Handle user login process.

        Returns:
            tuple: (is_logged_in, user_info)
        """
        if self.auth_method == "google" and self.authenticator:
            return self._google_login()
        else:
            return self._simple_login()

    def _google_login(self) -> Tuple[bool, Dict]:
        """Handle Google OAuth login."""
        try:
            # Check authentication status
            self.authenticator.check_authentification()

            # Show login button if not authenticated
            if not st.session_state.get('connected', False):
                self.authenticator.login()
                return False, {}

            # User is authenticated
            user_info = st.session_state.get('user_info', {})
            return True, {
                "email": user_info.get('email', ''),
                "name": user_info.get('name', 'User'),
                "picture": user_info.get('picture', ''),
                "oauth_id": st.session_state.get('oauth_id', ''),
                "auth_method": "google"
            }

        except Exception as e:
            logger.error(f"Google login error: {e}")
            return self._simple_login()

    def _simple_login(self) -> Tuple[bool, Dict]:
        """Handle simple username-based login."""
        if 'simple_auth_user' not in st.session_state:
            st.session_state.simple_auth_user = None

        if st.session_state.simple_auth_user:
            return True, {
                "email": f"{st.session_state.simple_auth_user}@local",
                "name": st.session_state.simple_auth_user,
                "picture": "",
                "oauth_id": st.session_state.simple_auth_user,
                "auth_method": "simple"
            }

        # Show simple login form
        with st.form("simple_login"):
            st.subheader("🔐 Welcome to EmpathyAI")
            st.write("Please enter a username to continue:")

            username = st.text_input("Username", placeholder="Enter any username...")

            if st.form_submit_button("Enter", use_container_width=True):
                if username.strip():
                    st.session_state.simple_auth_user = username.strip()
                    st.rerun()
                else:
                    st.error("Please enter a username")

        return False, {}

    def logout(self):
        """Handle user logout."""
        if self.auth_method == "google" and self.authenticator:
            try:
                self.authenticator.logout()
            except Exception as e:
                logger.error(f"Google logout error: {e}")

        # Clear simple auth
        if 'simple_auth_user' in st.session_state:
            del st.session_state.simple_auth_user

        # Clear other session data
        keys_to_clear = ['connected', 'user_info', 'oauth_id', 'conversation_history']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    def get_user_id(self, user_info: Dict) -> str:
        """
        Generate consistent user ID from user info.

        Args:
            user_info (dict): User information

        Returns:
            str: Unique user identifier
        """
        if user_info.get("auth_method") == "google":
            return user_info.get("oauth_id", user_info.get("email", "unknown"))
        else:
            return user_info.get("oauth_id", "unknown")

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        if self.auth_method == "google":
            return st.session_state.get('connected', False)
        else:
            return st.session_state.get('simple_auth_user') is not None

    def require_auth(self, redirect_message: str = "Please log in to continue."):
        """
        Require authentication to continue.

        Args:
            redirect_message (str): Message to show if not authenticated
        """
        is_logged_in, user_info = self.login()

        if not is_logged_in:
            st.info(redirect_message)
            st.stop()

        return user_info

# Global auth manager
_auth_manager = None

def get_auth_manager() -> AuthManager:
    """Get or create global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager

def login() -> Tuple[bool, Dict]:
    """
    Convenient function to handle login.

    Returns:
        tuple: (is_logged_in, user_info)
    """
    manager = get_auth_manager()
    return manager.login()

def logout():
    """Handle logout."""
    manager = get_auth_manager()
    manager.logout()

def require_authentication(message: str = "Please log in to access EmpathyAI.") -> Dict:
    """
    Require user authentication before proceeding.

    Args:
        message (str): Message to display if not authenticated

    Returns:
        dict: User information
    """
    manager = get_auth_manager()
    return manager.require_auth(message)

def get_current_user_id() -> Optional[str]:
    """Get current user ID if authenticated."""
    manager = get_auth_manager()
    if manager.is_authenticated():
        is_logged_in, user_info = manager.login()
        if is_logged_in:
            return manager.get_user_id(user_info)
    return None
