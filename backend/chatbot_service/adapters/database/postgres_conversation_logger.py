import logging
import asyncpg
from datetime import datetime, timezone
from chatbot_service.application.ports.conversation_logger_port import ConversationLoggerPort
from chatbot_service.core.configuration.config import settings
from chatbot_service.domain.exceptions import InfrastructureException

logger = logging.getLogger(__name__)

class PostgresConversationLogger(ConversationLoggerPort):
    def __init__(self, pool: asyncpg.Pool):
        """
        Initializes the logger with an asyncpg connection pool.
        The pool should be managed by the application's lifecycle (e.g., in main.py).
        """
        if pool is None:
            msg = "PostgresConversationLogger requires an initialized asyncpg.Pool."
            logger.critical(msg)
            raise InfrastructureException(msg)
        self._pool = pool
        logger.info("PostgresConversationLogger initialized with connection pool.")

    async def log_interaction(
        self,
        session_id: str,
        user_query: str,
        ai_response: str,
        timestamp: datetime,
        # context_used: str | None = None,
        # metadata: dict | None = None
    ) -> None:
        """
        Asynchronously logs a single user-AI interaction to PostgreSQL.
        Failures are logged but do not raise exceptions to block the main chat flow.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        query = """
            INSERT INTO chat_interactions (session_id, user_query, ai_response, created_at)
            VALUES ($1, $2, $3, $4)
        """
        # Add context_used and metadata to query and values if you include them
        # query = """
        #     INSERT INTO chat_interactions (session_id, user_query, ai_response, context_used, metadata, created_at)
        #     VALUES ($1, $2, $3, $4, $5, $6)
        # """
        # values = (session_id, user_query, ai_response, context_used, metadata, timestamp)

        values = (session_id, user_query, ai_response, timestamp)

        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(query, *values)
            logger.debug(f"Successfully logged interaction for session {session_id} to PostgreSQL.")
        except (asyncpg.PostgresError, OSError) as e:
            logger.error(
                f"Failed to log interaction for session {session_id} to PostgreSQL: {e}",
                exc_info=False
            )
        except Exception as e:
            logger.error(
                f"Unexpected error logging interaction for session {session_id} to PostgreSQL: {e}",
                exc_info=True
            )
