from abc import ABC, abstractmethod
from datetime import datetime

class ConversationLoggerPort(ABC):
    """
    Port defining the interface for logging chat conversations persistently.
    """
    @abstractmethod
    async def log_interaction(
        self,
        session_id: str,
        user_query: str,
        ai_response: str,
        timestamp: datetime,
        # Optional: context_used: str | None = None,
        # Optional: metadata: dict | None = None
    ) -> None:
        """
        Logs a single user-AI interaction.

        Args:
            session_id: The ID of the chat session.
            user_query: The user's input query.
            ai_response: The AI's generated response.
            timestamp: The UTC timestamp of the interaction.
            # context_used: Optional context string provided to the LLM.
            # metadata: Optional dictionary for any other relevant metadata.
        """
        pass