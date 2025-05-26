import logging
import asyncpg
from fastapi import FastAPI
from contextlib import asynccontextmanager
from chatbot_service.adapters.api.endpoints import chat_controller
from chatbot_service.core.configuration.config import settings
from chatbot_service.core.dependency_injection import Container, container, providers
from chatbot_service.core.configuration.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from chatbot_service.core.startup_checks import perform_startup_checks
from chatbot_service.domain.exceptions import InfrastructureException
import asyncio

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for application lifespan events.
    Handles startup and shutdown logic.
    """
    logger.info("Starting Chatbot RAG API...")
    app.state.container = container

    # --- Create PostgreSQL Connection Pool ---
    db_pool = None
    try:
        logger.info(f"Attempting to create PostgreSQL connection pool for DB: {settings.postgres_db}...")
        db_pool = await asyncpg.create_pool(
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            host=settings.postgres_host,
            port=settings.postgres_port,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size
        )
        container.adapters.postgres_pool.override(providers.Object(db_pool))
        logger.info("PostgreSQL connection pool created and DI provider updated.")
        app.state.db_pool = db_pool

    except (asyncpg.PostgresError, OSError) as e:
        logger.critical(f"FATAL STARTUP ERROR: Failed to create PostgreSQL pool: {e}", exc_info=True)
        container.adapters.postgres_pool.override(providers.Object(None))
        raise RuntimeError(f"Application startup failed: Could not connect to PostgreSQL - {e}") from e
    except Exception as e:
        logger.critical(f"FATAL STARTUP ERROR: Unexpected error creating PostgreSQL pool: {e}", exc_info=True)
        container.adapters.postgres_pool.override(providers.Object(None))
        raise RuntimeError(f"Application startup failed: Unexpected PostgreSQL setup error - {e}") from e
    try:
        loop = asyncio.get_running_loop()
        container.loop = loop

        await perform_startup_checks()

    except (InfrastructureException) as e:
        logger.critical(f"FATAL STARTUP ERROR: A critical dependency check failed after retries: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: Critical dependency error - {e}") from e
    except Exception as e:
        logger.critical(f"FATAL STARTUP ERROR: An unexpected error occurred during startup checks: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: Unexpected initialization error - {e}") from e
    finally:
        if hasattr(container, 'loop'):
            del container.loop

    logger.info("Application initialization complete. Ready to accept requests.")
    yield

    # --- Shutdown Logic ---
    logger.info("Executing application shutdown sequence...")
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        logger.info("Closing PostgreSQL connection pool...")
        try:
            await app.state.db_pool.close()
            logger.info("PostgreSQL connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}", exc_info=True)

    logger.info("Application shutdown complete.")

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application instance.
    """
    container.wire(modules=[
        "chatbot_service.adapters.api.endpoints.chat_controller",
        "chatbot_service.core.startup_checks"
    ])

    app = FastAPI(
        title="Chatbot RAG API",
        description="API for RAG Chatbot with Ollama, ChromaDB and Redis conversation memory",
        version="0.3.0",
        lifespan=lifespan
    )

    # Configure CORS middleware
    origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS middleware configured for origins: {origins}")

    # Include API routers
    app.include_router(chat_controller.router)
    logger.info("Included API routers.")

    return app

app = create_app()
