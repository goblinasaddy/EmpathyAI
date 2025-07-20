"""
Flexible memory management system supporting both SQLite and Google Sheets.
Handles user emotion history, conversation context, and analytics.
"""

import os
import sqlite3
import datetime
import logging
from typing import List, Dict, Optional, Union
import json

logger = logging.getLogger(__name__)

# Optional Google Sheets support
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    logger.info("Google Sheets not available - using SQLite only")

class MemoryManager:
    """Manages user memory using SQLite or Google Sheets backend."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.use_sheets = self._should_use_sheets()

        if self.use_sheets and SHEETS_AVAILABLE:
            self._init_sheets()
        else:
            self._init_sqlite()

    def _should_use_sheets(self) -> bool:
        """Determine whether to use Google Sheets based on environment."""
        try:
            # Check environment variable
            if os.getenv("USE_SHEETS", "").lower() == "true":
                return True

            # Check streamlit secrets
            import streamlit as st
            return st.secrets.get("USE_SHEETS", False)
        except:
            return False

    def _init_sqlite(self):
        """Initialize SQLite database."""
        try:
            self.db_path = "empathy_memory.db"
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._create_tables()
            logger.info(f"SQLite memory initialized for user: {self.user_id}")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            raise

    def _init_sheets(self):
        """Initialize Google Sheets connection."""
        try:
            credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "gcp_service_account.json")

            # Try streamlit secrets for credentials
            if not os.path.exists(credentials_path):
                try:
                    import streamlit as st
                    creds_dict = st.secrets["google_sheets_credentials"]

                    # Save to temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                        json.dump(dict(creds_dict), f)
                        credentials_path = f.name
                except:
                    raise Exception("Google Sheets credentials not found")

            scope = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
            self.gc = gspread.authorize(creds)

            sheet_name = os.getenv("SHEET_NAME", "EmpathyAI_Memory")
            try:
                # Try streamlit secrets
                import streamlit as st
                sheet_name = st.secrets.get("SHEET_NAME", sheet_name)
            except:
                pass

            self.sheet = self.gc.open(sheet_name).worksheet("emotions")
            logger.info(f"Google Sheets memory initialized for user: {self.user_id}")

        except Exception as e:
            logger.error(f"Google Sheets initialization failed: {e}")
            # Fallback to SQLite
            self.use_sheets = False
            self._init_sqlite()

    def _create_tables(self):
        """Create necessary SQLite tables."""
        # Create emotions table
        emotions_table = """CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            emotion_label TEXT NOT NULL,
            confidence REAL,
            message_text TEXT,
            response_text TEXT,
            session_id TEXT
        )"""

        # Create user profiles table  
        profiles_table = """CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            total_conversations INTEGER DEFAULT 0,
            preferred_name TEXT,
            timezone TEXT,
            last_active TEXT
        )"""

        # Create conversation context table
        context_table = """CREATE TABLE IF NOT EXISTS conversation_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            message_pair TEXT,
            timestamp TEXT NOT NULL
        )"""

        tables = [emotions_table, profiles_table, context_table]

        for table_sql in tables:
            self.conn.execute(table_sql)
        self.conn.commit()

    def add_emotion_record(self, emotion_label: str, confidence: float = None, 
                          message: str = None, response: str = None, 
                          session_id: str = None) -> bool:
        """Add a new emotion record to memory."""
        try:
            timestamp = datetime.datetime.utcnow().isoformat()

            if self.use_sheets:
                return self._add_to_sheets(timestamp, emotion_label, confidence, 
                                         message, response, session_id)
            else:
                return self._add_to_sqlite(timestamp, emotion_label, confidence,
                                         message, response, session_id)
        except Exception as e:
            logger.error(f"Failed to add emotion record: {e}")
            return False

    def _add_to_sheets(self, timestamp: str, emotion: str, confidence: float,
                      message: str, response: str, session_id: str) -> bool:
        """Add record to Google Sheets."""
        try:
            row = [
                timestamp, self.user_id, emotion, confidence or 0.5,
                message or "", response or "", session_id or ""
            ]
            self.sheet.append_row(row)
            return True
        except Exception as e:
            logger.error(f"Failed to add to sheets: {e}")
            return False

    def _add_to_sqlite(self, timestamp: str, emotion: str, confidence: float,
                      message: str, response: str, session_id: str) -> bool:
        """Add record to SQLite."""
        try:
            insert_query = """INSERT INTO emotions 
                (user_id, timestamp, emotion_label, confidence, 
                 message_text, response_text, session_id) 
                VALUES (?, ?, ?, ?, ?, ?, ?)"""

            self.conn.execute(insert_query, 
                (self.user_id, timestamp, emotion, confidence or 0.5,
                 message, response, session_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add to SQLite: {e}")
            return False

    def get_recent_emotions(self, limit: int = 30) -> List[Dict]:
        """Get recent emotion records for the user."""
        try:
            if self.use_sheets:
                return self._get_from_sheets(limit)
            else:
                return self._get_from_sqlite(limit)
        except Exception as e:
            logger.error(f"Failed to get recent emotions: {e}")
            return []

    def _get_from_sheets(self, limit: int) -> List[Dict]:
        """Get records from Google Sheets."""
        try:
            all_records = self.sheet.get_all_records()
            user_records = [r for r in all_records if r.get("user_id") == self.user_id]

            # Sort by timestamp (most recent first)
            sorted_records = sorted(
                user_records, 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )[:limit]

            return [
                {
                    "timestamp": r.get("timestamp"),
                    "emotion": r.get("emotion_label"),
                    "confidence": float(r.get("confidence", 0.5)),
                    "message": r.get("message_text", ""),
                    "session_id": r.get("session_id", "")
                }
                for r in sorted_records
            ]
        except Exception as e:
            logger.error(f"Failed to get from sheets: {e}")
            return []

    def _get_from_sqlite(self, limit: int) -> List[Dict]:
        """Get records from SQLite."""
        try:
            select_query = """SELECT timestamp, emotion_label, confidence, message_text, session_id
                FROM emotions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?"""

            cursor = self.conn.execute(select_query, (self.user_id, limit))

            records = []
            for row in cursor.fetchall():
                records.append({
                    "timestamp": row[0],
                    "emotion": row[1],
                    "confidence": row[2],
                    "message": row[3] or "",
                    "session_id": row[4] or ""
                })

            return records
        except Exception as e:
            logger.error(f"Failed to get from SQLite: {e}")
            return []

    def get_emotion_patterns(self, days: int = 7) -> Dict:
        """Analyze emotion patterns over the specified time period."""
        try:
            cutoff_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()

            if self.use_sheets:
                records = self._get_from_sheets(1000)  # Get more records for analysis
                # Filter by date
                recent_records = [
                    r for r in records 
                    if r["timestamp"] >= cutoff_date
                ]
            else:
                pattern_query = """SELECT emotion_label, confidence, timestamp
                    FROM emotions 
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC"""

                cursor = self.conn.execute(pattern_query, (self.user_id, cutoff_date))

                recent_records = [
                    {"emotion": row[0], "confidence": row[1], "timestamp": row[2]}
                    for row in cursor.fetchall()
                ]

            if not recent_records:
                return {"total_entries": 0, "patterns": {}}

            # Analyze patterns
            emotion_counts = {}
            total_confidence = 0

            for record in recent_records:
                emotion = record["emotion"]
                confidence = record.get("confidence", 0.5)

                if emotion not in emotion_counts:
                    emotion_counts[emotion] = {"count": 0, "total_confidence": 0}

                emotion_counts[emotion]["count"] += 1
                emotion_counts[emotion]["total_confidence"] += confidence
                total_confidence += confidence

            # Calculate averages
            patterns = {}
            for emotion, data in emotion_counts.items():
                patterns[emotion] = {
                    "frequency": data["count"],
                    "percentage": round((data["count"] / len(recent_records)) * 100, 1),
                    "avg_confidence": round(data["total_confidence"] / data["count"], 2)
                }

            return {
                "total_entries": len(recent_records),
                "avg_confidence": round(total_confidence / len(recent_records), 2),
                "patterns": patterns,
                "time_period": f"{days} days"
            }

        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return {"total_entries": 0, "patterns": {}}

    def close(self):
        """Close database connections."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


# Factory function
def create_memory_manager(user_id: str) -> MemoryManager:
    """Create a memory manager for the given user."""
    return MemoryManager(user_id)
