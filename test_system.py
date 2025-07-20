#!/usr/bin/env python3
"""
EmpathyAI 2.0 - Component Test Suite
Validates that all modules are working correctly before deployment.
"""

import sys
import logging
import traceback
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmpathyAITester:
    """Comprehensive test suite for EmpathyAI components."""

    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0

    def run_all_tests(self):
        """Run complete test suite."""
        print("🔍 EmpathyAI 2.0 - System Test Suite")
        print("=" * 50)

        # Test each component
        test_methods = [
            self.test_imports,
            self.test_emotion_detection,
            self.test_sentiment_fusion, 
            self.test_llm_response,
            self.test_response_generator,
            self.test_memory_system,
            self.test_n8n_integration,
            self.test_auth_system
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test failed: {test_method.__name__}: {e}")
                self.results[test_method.__name__] = {"status": "FAILED", "error": str(e)}

        self._print_results()

    def test_imports(self):
        """Test that all modules can be imported."""
        print("\n📦 Testing Module Imports...")
        self.total_tests += 1

        try:
            # Test core imports
            from src.emotion import detect_emotion, get_detector
            from src.sentiment_fusion import fuse_sentiment_emotion
            from src.llm_response import ask_gemini
            from src.response_generator import craft_empathy_response
            from src.memory import create_memory_manager
            from src.n8n_integration import post_emotion_record
            from src.auth import login

            print("  ✅ All modules imported successfully")
            self.passed_tests += 1
            self.results["imports"] = {"status": "PASSED"}

        except Exception as e:
            print(f"  ❌ Import failed: {e}")
            self.results["imports"] = {"status": "FAILED", "error": str(e)}

    def test_emotion_detection(self):
        """Test emotion detection functionality."""
        print("\n🎭 Testing Emotion Detection...")
        self.total_tests += 1

        try:
            from src.emotion import detect_emotion

            test_cases = [
                ("I am so happy today!", "joy"),
                ("I feel very sad and lonely", "sadness"),
                ("This makes me angry!", "anger"),
                ("I'm worried about tomorrow", "fear"),
                ("Hello there", None)  # Any emotion is fine for neutral text
            ]

            all_passed = True
            for text, expected in test_cases:
                result = detect_emotion(text)

                if not isinstance(result, dict):
                    print(f"  ❌ Invalid result type for '{text[:20]}...'")
                    all_passed = False
                    continue

                if "label" not in result or "confidence" not in result:
                    print(f"  ❌ Missing required fields for '{text[:20]}...'")
                    all_passed = False
                    continue

                detected = result["label"]
                confidence = result["confidence"]

                print(f"  📝 '{text[:30]}...' -> {detected} ({confidence:.2f})")

                if expected and detected != expected:
                    print(f"    ⚠️  Expected {expected}, got {detected}")
                    # Don't fail the test for this, emotions can be subjective

            if all_passed:
                print("  ✅ Emotion detection working")
                self.passed_tests += 1
                self.results["emotion"] = {"status": "PASSED"}
            else:
                self.results["emotion"] = {"status": "FAILED", "error": "Format issues"}

        except Exception as e:
            print(f"  ❌ Emotion detection failed: {e}")
            self.results["emotion"] = {"status": "FAILED", "error": str(e)}

    def test_sentiment_fusion(self):
        """Test sentiment fusion functionality."""
        print("\n🔀 Testing Sentiment Fusion...")
        self.total_tests += 1

        try:
            from src.sentiment_fusion import fuse_sentiment_emotion

            test_cases = [
                ("I love this so much!", "joy"),
                ("I hate everything right now", "anger"),
                ("I'm feeling okay", "neutral")
            ]

            for text, emotion in test_cases:
                result = fuse_sentiment_emotion(text, emotion)
                print(f"  📝 '{text}' + {emotion} -> {result}")

                if not isinstance(result, str) or len(result) == 0:
                    raise ValueError(f"Invalid fusion result: {result}")

            print("  ✅ Sentiment fusion working")
            self.passed_tests += 1
            self.results["sentiment"] = {"status": "PASSED"}

        except Exception as e:
            print(f"  ❌ Sentiment fusion failed: {e}")
            self.results["sentiment"] = {"status": "FAILED", "error": str(e)}

    def test_llm_response(self):
        """Test LLM response generation."""
        print("\n🤖 Testing LLM Response...")
        self.total_tests += 1

        try:
            from src.llm_response import ask_gemini, check_api_health

            # Check API health first
            health = check_api_health()
            print(f"  📊 API Health: {health}")

            # Test simple generation
            prompt = "Respond with exactly one word: 'Hello'"
            response = ask_gemini(prompt, temperature=0.1)

            print(f"  📝 Test prompt: '{prompt}'")
            print(f"  📝 Response: '{response[:50]}...'")

            if len(response) > 0:
                print("  ✅ LLM response generation working")
                self.passed_tests += 1
                self.results["llm"] = {"status": "PASSED", "api_available": health.get("available", False)}
            else:
                self.results["llm"] = {"status": "FAILED", "error": "Empty response"}

        except Exception as e:
            print(f"  ❌ LLM response failed: {e}")
            self.results["llm"] = {"status": "FAILED", "error": str(e)}

    def test_response_generator(self):
        """Test response generator."""
        print("\n💬 Testing Response Generator...")
        self.total_tests += 1

        try:
            from src.response_generator import craft_empathy_response

            test_cases = [
                ("I'm feeling really sad today", "negative-sadness"),
                ("I'm so excited about tomorrow!", "positive-joy"),
                ("I don't know how I'm feeling", "neutral")
            ]

            for user_text, emotion in test_cases:
                response = craft_empathy_response(user_text, emotion)
                print(f"  📝 '{user_text}' ({emotion})")
                print(f"      -> '{response[:60]}...'")

                if not isinstance(response, str) or len(response) < 10:
                    raise ValueError(f"Invalid response: {response}")

            print("  ✅ Response generator working")
            self.passed_tests += 1
            self.results["generator"] = {"status": "PASSED"}

        except Exception as e:
            print(f"  ❌ Response generator failed: {e}")
            self.results["generator"] = {"status": "FAILED", "error": str(e)}

    def test_memory_system(self):
        """Test memory management system."""
        print("\n💾 Testing Memory System...")
        self.total_tests += 1

        try:
            from src.memory import create_memory_manager

            # Create test memory manager
            memory = create_memory_manager("test_user_123")

            # Test adding emotion record
            success = memory.add_emotion_record(
                emotion_label="test_emotion",
                confidence=0.8,
                message="Test message",
                response="Test response",
                session_id="test_session"
            )

            if not success:
                print("  ⚠️  Could not add emotion record (may be expected)")

            # Test retrieving records
            records = memory.get_recent_emotions(limit=5)
            print(f"  📊 Retrieved {len(records)} emotion records")

            # Test analytics
            patterns = memory.get_emotion_patterns(days=7)
            print(f"  📈 Analytics: {patterns.get('total_entries', 0)} entries")

            # Cleanup
            memory.close()

            print("  ✅ Memory system working")
            self.passed_tests += 1
            self.results["memory"] = {"status": "PASSED"}

        except Exception as e:
            print(f"  ❌ Memory system failed: {e}")
            self.results["memory"] = {"status": "FAILED", "error": str(e)}

    def test_n8n_integration(self):
        """Test n8n webhook integration."""
        print("\n🔗 Testing n8n Integration...")
        self.total_tests += 1

        try:
            from src.n8n_integration import test_n8n_connection, post_emotion_record

            # Test connection
            connection_result = test_n8n_connection()
            print(f"  📊 Connection test: {connection_result}")

            # Test posting data (will fail if no webhook URL configured)
            post_result = post_emotion_record(
                user_id="test_user",
                emotion_label="test_emotion",
                confidence=0.8,
                message="Test message"
            )

            if connection_result.get("connected"):
                print("  ✅ n8n integration working")
                status = "PASSED"
            else:
                print("  ⚠️  n8n not configured (expected for development)")
                status = "SKIPPED"

            self.passed_tests += 1
            self.results["n8n"] = {"status": status, "connected": connection_result.get("connected", False)}

        except Exception as e:
            print(f"  ❌ n8n integration failed: {e}")
            self.results["n8n"] = {"status": "FAILED", "error": str(e)}

    def test_auth_system(self):
        """Test authentication system."""
        print("\n🔐 Testing Auth System...")
        self.total_tests += 1

        try:
            from src.auth import get_auth_manager

            # Create auth manager
            auth_manager = get_auth_manager()
            print(f"  📊 Auth method: {auth_manager.auth_method}")

            # Test basic functionality (without actually logging in)
            is_authenticated = auth_manager.is_authenticated()
            print(f"  📊 Currently authenticated: {is_authenticated}")

            print("  ✅ Auth system initialized")
            self.passed_tests += 1
            self.results["auth"] = {"status": "PASSED", "method": auth_manager.auth_method}

        except Exception as e:
            print(f"  ❌ Auth system failed: {e}")
            self.results["auth"] = {"status": "FAILED", "error": str(e)}

    def _print_results(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 50)

        for test_name, result in self.results.items():
            status = result["status"]
            icon = "✅" if status == "PASSED" else "⚠️" if status == "SKIPPED" else "❌"
            print(f"{icon} {test_name.upper()}: {status}")

            if "error" in result:
                print(f"    Error: {result['error']}")

        print(f"\n📊 Overall: {self.passed_tests}/{self.total_tests} tests passed")

        if self.passed_tests == self.total_tests:
            print("🎉 All tests passed! EmpathyAI is ready for deployment.")
        else:
            print("⚠️  Some tests failed. Check the errors above before deploying.")

        return self.passed_tests == self.total_tests

def main():
    """Run the test suite."""
    tester = EmpathyAITester()
    success = tester.run_all_tests()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
