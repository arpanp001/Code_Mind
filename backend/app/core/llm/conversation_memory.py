# backend/app/core/llm/conversation_memory.py
#
# Manages per-session conversation history.
#
# How it works:
#   Each browser session gets a session_id (UUID generated on frontend).
#   We store the last N message pairs for that session.
#   These are injected into the Gemini prompt as "previous conversation".
#
# Storage: in-memory dict (good enough for development).
# For production: replace with Redis.

import time
from collections import deque
from dataclasses import dataclass, field
from typing      import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum message pairs to remember per session
MAX_HISTORY_PAIRS = 5

# Session expires after this many seconds of inactivity (1 hour)
SESSION_TTL = 3600


@dataclass
class Message:
    """A single message in the conversation."""
    role:      str    # "user" or "assistant"
    content:   str
    timestamp: float  = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """A conversation session with message history."""
    session_id:   str
    project_id:   str
    messages:     deque = field(
        default_factory=lambda: deque(maxlen=MAX_HISTORY_PAIRS * 2)
    )
    created_at:   float = field(default_factory=time.time)
    last_active:  float = field(default_factory=time.time)

    def add_exchange(self, question: str, answer: str) -> None:
        """Adds a question-answer pair to the history."""
        self.messages.append(Message(role="user",      content=question))
        self.messages.append(Message(role="assistant", content=answer))
        self.last_active = time.time()

    def get_history_text(self) -> str:
        """
        Formats conversation history for injection into the prompt.
        Returns empty string if no history.
        """
        if not self.messages:
            return ""

        lines = ["Previous conversation:"]
        for msg in self.messages:
            prefix = "User"      if msg.role == "user"      else "Assistant"
            # Truncate long messages to keep prompt size manageable
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

    def is_expired(self) -> bool:
        """Returns True if the session has been inactive too long."""
        return (time.time() - self.last_active) > SESSION_TTL


class ConversationMemoryManager:
    """
    Manages conversation sessions in memory.

    Usage:
        manager  = ConversationMemoryManager()
        session  = manager.get_or_create("session_abc", "proj_xyz")
        history  = session.get_history_text()
        # ... generate answer ...
        session.add_exchange(question, answer)
    """

    def __init__(self):
        # session_id → ConversationSession
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create(
        self,
        session_id: str,
        project_id: str,
    ) -> ConversationSession:
        """
        Returns existing session or creates a new one.
        Also cleans up expired sessions opportunistically.
        """
        # Periodic cleanup (every ~100 calls)
        if len(self._sessions) % 100 == 0:
            self._cleanup_expired()

        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id = session_id,
                project_id = project_id,
            )
            logger.debug(f"New session: {session_id[:8]}... [{project_id}]")

        session = self._sessions[session_id]
        session.last_active = time.time()
        return session

    def clear_session(self, session_id: str) -> None:
        """Clears a session's history (user clicked 'Clear chat')."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Cleared session: {session_id[:8]}...")

    def _cleanup_expired(self) -> None:
        """Removes expired sessions to prevent memory leaks."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def get_stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "max_history_pairs": MAX_HISTORY_PAIRS,
        }


# Module-level singleton
conversation_memory = ConversationMemoryManager()