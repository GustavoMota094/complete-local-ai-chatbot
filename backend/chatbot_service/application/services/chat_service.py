import asyncio
import logging
from datetime import datetime, timezone
from typing import List
from langchain.memory import ConversationBufferWindowMemory
from langchain_redis.chat_message_history import RedisChatMessageHistory
from langchain_core.documents import Document
from chatbot_service.application.ports.intent_port import IntentPort
from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.application.ports.llm_port import LLMPort
from chatbot_service.application.ports.conversation_logger_port import ConversationLoggerPort
from chatbot_service.domain.exceptions import ApplicationException, IndexNotReadyError
from chatbot_service.core.configuration.config import settings
from chatbot_service.core.utils import create_safe_redis_identifier
import redis

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(
        self, 
        vector_store: VectorStorePort, 
        llm_client: LLMPort, 
        conversation_logger: ConversationLoggerPort,
        intent_client: IntentPort
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.conversation_logger = conversation_logger
        self.intent_client = intent_client
        logger.info("ChatService initialized.")

        try:
            test_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_connect_timeout=2
            )
            test_client.ping()
            logger.info("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis on initialization: {e}", exc_info=True)

    # Initialize memory components for a session
    def _get_chat_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        safe_redis_id = create_safe_redis_identifier(session_id)
        redis_key = f"{settings.redis_session_prefix}{safe_redis_id}"
        logger.debug(f"Using Redis key '{redis_key}' for session {session_id} with TTL {settings.redis_session_ttl}s")

        try:
            redis_history = RedisChatMessageHistory(
                session_id=redis_key,
                redis_url=f"redis://{':'+settings.redis_password+'@' if settings.redis_password else ''}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                ttl=settings.redis_session_ttl
            )

            memory = ConversationBufferWindowMemory(
                k=settings.memory_window_size,
                chat_memory=redis_history,
                return_messages=False,
                memory_key="chat_history",
                input_key="question",
                output_key="output"
            )
            return memory

        except redis.exceptions.ConnectionError as e:
             logger.error(f"Failed to connect to Redis when creating history for session {session_id}: {e}", exc_info=True)
             raise ApplicationException("Could not connect to the message store.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while initializing memory for session {session_id}: {e}", exc_info=True)
            raise ApplicationException("An error occurred while setting up chat memory.") from e

    # Generate the chat response
    async def generate_chat_response(self, query: str, session_id: str) -> str:
        logger.info(f"Processing query for session_id: {session_id}")
        try:
            # --- Check Vector Store Readiness ---
            if not self.vector_store.is_ready():
                 logger.error("Vector store is not ready.")
                 raise IndexNotReadyError("Vector store is not initialized properly.")

            # --- Intent Classification ---
            detected_intent = await self.intent_client.classify_intent(query)
            logger.info(f"Detected intent for query '{query[:50]}...': {detected_intent} (Session: {session_id})")

            # --- Memory Management ---
            memory = self._get_chat_memory(session_id)

            memory_dict = memory.load_memory_variables({})
            chat_history_string = memory_dict.get("chat_history", "")
            logger.debug(f"Loaded history string via Redis for session {session_id}: \n{chat_history_string}")

            # --- RAG Search ---
            logger.info(f"Searching vector store for query: '{query[:50]}...' with k={settings.rag_search_k}")
            raw_relevant_documents: List[Document] = self.vector_store.search(
                query,
                k=settings.rag_search_k
            )
            
            # --- Relevance Filtering ---
            filtered_documents: List[Document] = []
            if raw_relevant_documents: # Check if there are documents to process
                logger.debug(f"Raw search results count: {len(raw_relevant_documents)}")

                # --- Loop through the raw documents to filter them ---
                for i, doc in enumerate(raw_relevant_documents): # Iterate through the non-empty list
                    distance = doc.metadata.get('distance')
                    source = doc.metadata.get('source', 'N/A')
                    score_str = "N/A"
                    should_append = False # Flag to track if we should append this doc

                    if distance is not None:
                        similarity_score = 1.0 - distance
                        score_str = f"{similarity_score:.4f}"
                        # Log score information for this document
                        logger.debug(f"  Result {i+1}: Source='{source}', Distance={distance:.4f}, Calc. Similarity={score_str}")
                        logger.debug(f"--- Current Threshold: {settings.rag_relevance_score_threshold:.4f} ---")

                        # --- Perform the actual filtering check ---
                        if similarity_score >= settings.rag_relevance_score_threshold:
                            logger.debug(f"PASSED THRESHOLD! Including doc: {source}")
                            should_append = True
                        else:
                             logger.debug(f"Filtering out doc below threshold ({settings.rag_relevance_score_threshold:.2f})")
                    else:
                         # Handle missing distance - decide whether to include or not
                         logger.warning(f"Document {source} found without distance metadata. Including by default.")
                         should_append = True # Or set to False if you want to exclude docs without distance

                    # Append if the flag is set
                    if should_append:
                        filtered_documents.append(doc)

            logger.info(f"Found {len(filtered_documents)} relevant chunks after filtering.")

            # --- Context String Construction ---
            if filtered_documents:
                context_chunks = [doc.page_content for doc in filtered_documents]
                context_string = "\n\n".join(context_chunks)
                logger.debug(f"Context string generated from filtered docs: \n{context_string[:500]}...")
            else:
                context_string = "No relevant context found in the knowledge base."
                logger.info("No relevant chunks found after filtering.")

            # --- LLM Call ---
            logger.info("Generating response with LLM client.")
            current_utc_time = datetime.now(timezone.utc)
            response = await self.llm_client.generate_response(
                query=query,
                context=context_string,
                chat_history=chat_history_string
            )
            logger.info(f"LLM generated response successfully for session {session_id}.")
            logger.debug(f"LLM Response: {response}")

            # --- Save interaction to memory ---
            memory.save_context({"question": query}, {"output": response})
            logger.info(f"Saved interaction to Redis memory for session {session_id}")

            # --- Asynchronously log interaction to PostgreSQL ---
            asyncio.create_task(
                self.conversation_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    ai_response=response,
                    timestamp=current_utc_time
                    # context_used=context_string,
                    # metadata={"some_key": "some_value"}
                )
            )
            logger.debug(f"Scheduled background task for PostgreSQL logging for session {session_id}")

            return response

        except IndexNotReadyError as e:
             logger.error(f"IndexNotReadyError encountered for session {session_id}: {e}", exc_info=True)
             raise ApplicationException("The document index is not ready. Please wait or contact support.") from e
        except redis.exceptions.ConnectionError as e:
             logger.error(f"Redis connection error during chat generation for session {session_id}: {e}", exc_info=True)
             raise ApplicationException("Failed to communicate with the message store.") from e
        except Exception as e:
             logger.error(f"An unexpected error occurred in chat service for session {session_id}: {e}", exc_info=True)
             raise ApplicationException("An error occurred while processing your request.") from e

    # Clear chat history for a session
    def clear_chat_history(self, session_id: str) -> None:
        safe_redis_id = create_safe_redis_identifier(session_id)
        logger.info(f"Attempting to clear chat history for session_id: {session_id}")
        redis_key = f"{settings.redis_session_prefix}{safe_redis_id}"
        try:
            redis_history = RedisChatMessageHistory(
                session_id=redis_key,
                redis_url=f"redis://{':'+settings.redis_password+'@' if settings.redis_password else ''}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            )
            redis_history.clear()
            logger.info(f"Successfully cleared chat history for Redis key: {redis_key}")

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error while trying to clear history for session {session_id}: {e}", exc_info=True)
            raise ApplicationException("Failed to connect to the message store to clear history.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while clearing history for session {session_id}: {e}", exc_info=True)
            raise ApplicationException("An error occurred while clearing chat history.") from e
    