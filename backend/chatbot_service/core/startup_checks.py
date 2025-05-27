import asyncpg
import logging
import time
import uuid
from dependency_injector.wiring import Provide, inject
from redis.exceptions import ConnectionError, TimeoutError
from langchain_redis.chat_message_history import RedisChatMessageHistory

from chatbot_service.core.dependency_injection import Container
from chatbot_service.core.configuration.config import settings
from chatbot_service.domain.exceptions import InfrastructureException, IndexNotReadyError
from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.application.ports.llm_port import LLMPort
# from chatbot_service.application.ports.auth_service_port import AuthServicePort # Future import
from chatbot_service.application.services.chat_service import ChatService

logger = logging.getLogger(__name__)

# --- Configuration for Retries ---
RETRY_ATTEMPTS = settings.startup_check_retry_attempts
RETRY_DELAY_SECONDS = settings.startup_check_retry_delay_seconds

# --- Helper for Retries ---
def _retry_check(check_func, check_name: str, *args, **kwargs):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            logger.debug(f"Attempt {attempt}/{RETRY_ATTEMPTS} for check: {check_name}")
            result = check_func(*args, **kwargs)
            logger.info(f"Check successful: {check_name}")
            return result
        except (ConnectionError, TimeoutError, IndexNotReadyError, InfrastructureException) as e:
            logger.warning(f"Check failed for {check_name} on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if attempt == RETRY_ATTEMPTS:
                logger.critical(f"Check failed for {check_name} after {RETRY_ATTEMPTS} attempts.")
                raise InfrastructureException(f"Failed check '{check_name}' after {RETRY_ATTEMPTS} attempts") from e
            logger.info(f"Retrying {check_name} in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
    raise InfrastructureException(f"Check '{check_name}' unexpectedly finished retry loop.")


# --- Individual Check Functions ---
@inject
def check_vector_store(
    vector_store: VectorStorePort = Provide[Container.adapters.vector_store]
):
    """Checks if the vector store is initialized and ready."""
    initial_count = vector_store.get_collection_count()
    logger.debug(f"Vector Store check via count: {initial_count}")

@inject
def check_llm_client(
    llm_client: LLMPort = Provide[Container.adapters.llm_client]
):
    """Checks if the LLM client was initialized (e.g., template loaded)."""
    logger.debug("LLM Client check passed (assumes __init__ succeeded).")


@inject
def check_redis_via_chat_service(
    chat_service: ChatService = Provide[Container.application.chat_service]
):
    """Checks Redis connection via the ChatService initialization check."""
    logger.debug("Redis check via Chat Service initialization passed.")

# --- Placeholder for Future Authentication Check ---
@inject
def check_auth_service(
    # auth_client: AuthServicePort = Provide[Container.adapters.auth_service] # Example
):
    """Placeholder for checking the authentication service."""
    logger.debug("Auth Service check: Placeholder - Not implemented.")
    # Example implementation:
    # try:
    #     status = auth_client.ping() # Assuming an AuthServicePort and adapter exist
    #     if not status:
    #         raise InfrastructureException("Auth service ping returned unhealthy status.")
    # except Exception as e:
    #     raise InfrastructureException("Failed to connect or verify Auth service.") from e
    pass # Remove pass when implemented


# --- End-to-End Chatbot Check ---
@inject
async def check_chatbot_e2e(
    llm_client: LLMPort = Provide[Container.adapters.llm_client]
    # vector_store: VectorStorePort = Provide[Container.adapters.vector_store] # Optional: if checking search
):
    """
    Performs a lightweight E2E check by sending a test message to the LLM.
    Verifies LLM connectivity and basic response generation.
    Optionally, can be extended to check vector search and Redis save/delete.
    """
    test_query = "Startup Check: Respond with 'OK'"
    test_context = "This is a system startup test."
    test_history = ""
    test_intent = "Test"

    logger.debug("Performing E2E check: Sending test query to LLM...")
    try:
        response = await llm_client.generate_response(
            query=test_query,
            context=test_context,
            chat_history=test_history,
            intent=test_intent
        )

        if not response or not isinstance(response, str):
            raise InfrastructureException(f"E2E check failed: LLM returned invalid response type ({type(response)}) or empty response.")

        logger.debug(f"E2E check: LLM response received: '{response[:100]}...'")

        if "error" in response.lower() or len(response) < 1:
             logger.warning(f"E2E check warning: LLM response might indicate an issue: {response}")
             raise InfrastructureException(f"E2E check failed: LLM response indicates potential error: {response}")

        logger.debug("E2E check basic validation passed.")

    except Exception as e:
        logger.error(f"E2E check failed during execution: {e}", exc_info=True)
        raise InfrastructureException(f"E2E check failed: {e}") from e


# --- Orchestrator Function ---

@inject
async def perform_startup_checks(container: Container = Provide[Container]):
    """Runs all critical startup dependency checks, applying retries where appropriate."""
    logger.info("Starting critical dependency checks...")

    # --- Perform checks ---

    # Vector Store Check (Retryable)
    await container.loop.run_in_executor(None, _retry_check, check_vector_store, "Vector Store Readiness")

    # LLM Client Init Check (Sync)
    await container.loop.run_in_executor(None, check_llm_client)

    # Redis Check (via ChatService Init) (Retryable - Sync)
    await container.loop.run_in_executor(None, _retry_check, check_redis_via_chat_service, "Redis Connection (via ChatService)")

    # Auth Service Check (Sync Placeholder)
    await container.loop.run_in_executor(None, check_auth_service)

    # E2E Check (Async)
    await _retry_check(check_chatbot_e2e, "Chatbot E2E Check")

    logger.info("All critical dependency checks passed.")
