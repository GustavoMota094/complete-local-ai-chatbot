import os
import logging
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """
    # --- Logging Configuration ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    # log_file_path: str = os.getenv("LOG_FILE_PATH", "chatbot_service.log") # Optional: For file logging

    # --- Startup Check Configuration ---
    startup_check_retry_attempts: int = int(os.getenv("STARTUP_CHECK_RETRY_ATTEMPTS", 5))
    startup_check_retry_delay_seconds: int = int(os.getenv("STARTUP_CHECK_RETRY_DELAY_SECONDS", 5))

    # --- Ollama/ Main LLM Configuration ---
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    system_message: str = os.getenv("SYSTEM_MESSAGE", "Voce e um assistente de suporte TI e deve ajudar solicitantes usando sua base de conhecimento e os documentos fornecidos.")
    template_file_path: Path = Path(os.getenv("TEMPLATE_FILE_PATH", "./templates/ollama_prompt_template.txt"))

    # --- Intent Classification Configuration ---
    ollama_intent_model: str = os.getenv("OLLAMA_INTENT_MODEL", "llama3")
    intent_system_message: str = os.getenv("INTENT_SYSTEM_MESSAGE", "Voce e um classificador de intencao de suporte TI. Classifique a pergunta do usuario em uma das seguintes categorias:")
    intent_categories: str = os.getenv("INTENT_CATEGORIES", "Pergunta Geral, Saudacao e Despedida.")
    intent_template_file_path: Path = Path(os.getenv("INTENT_TEMPLATE_FILE_PATH", "./templates/intent_prompt_template.txt"))
    default_intent: str = os.getenv("DEFAULT_INTENT", "Duvidas Gerais")

    # --- Vector Store Configuration ---
    vector_store_type: str = "chroma"
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    chroma_persist_directory: Path = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    # --- RAG Configuration ---
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", 1000)) # Target chunk size for splitting
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", 150)) # Overlap between chunks
    rag_search_k: int = int(os.getenv("RAG_SEARCH_K", 5)) # Number of relevant documents to retrieve from vector store
    rag_relevance_score_threshold: float = float(os.getenv("RAG_RELEVANCE_SCORE_THRESHOLD", 0.70)) # (0.0 to 1.0)

    # --- Memory Configuration ---
    # Number of past interactions to keep in conversation memory
    memory_window_size: int = int(os.getenv("MEMORY_WINDOW_SIZE", 5))

    # --- Redis Chat History Configuration ---
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_db: int = int(os.getenv("REDIS_DB", 0))
    redis_password: str | None = os.getenv("REDIS_PASSWORD")
    redis_session_ttl: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", 3600))
    redis_session_prefix: str = os.getenv("REDIS_SESSION_PREFIX", "chat_session:")

    # --- PostgreSQL Chat Logging Configuration ---
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_user: str = os.getenv("POSTGRES_USER", "your_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "your_password")
    postgres_db: str = os.getenv("POSTGRES_DB", "chatbot_logs")
    postgres_pool_min_size: int = int(os.getenv("POSTGRES_POOL_MIN_SIZE", 1))
    postgres_pool_max_size: int = int(os.getenv("POSTGRES_POOL_MAX_SIZE", 10))

    # --- Document Paths ---
    md_documents_path: Path = Path(os.getenv("MD_DOCUMENTS_PATH", "./data/md_documents"))

    # --- API Configuration ---
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")


    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

# --- Post-initialization actions ---

logger.info(f"Log Level set to: {settings.log_level}")
logger.info(f"Ollama Model: {settings.ollama_model} at {settings.ollama_base_url}")
logger.info(f"Embedding Model: {settings.embeddings_model_name}")
logger.info(f"Vector Store: ChromaDB at {settings.chroma_persist_directory}, Collection: {settings.chroma_collection_name}")
logger.info(f"Redis configured: Host={settings.redis_host}, Port={settings.redis_port}, DB={settings.redis_db}, TTL={settings.redis_session_ttl}s")
logger.info(f"PostgreSQL configured for logging: Host={settings.postgres_host}, DB={settings.postgres_db}")
logger.info(f"Markdown documents path: {settings.md_documents_path}")
logger.info(f"API Base URL: {settings.api_base_url}")

# Ensure necessary directories exist
try:
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    logger.info(f"Ensured ChromaDB persistence directory exists: {settings.chroma_persist_directory}")
    os.makedirs(settings.md_documents_path, exist_ok=True)
    logger.info(f"Ensured Markdown documents path exists: {settings.md_documents_path}")
    if settings.template_file_path.parent != Path("."):
         os.makedirs(settings.template_file_path.parent, exist_ok=True)
         logger.info(f"Ensured template directory exists: {settings.template_file_path.parent}")

except OSError as e:
    logger.error(f"Error creating necessary directories: {e}", exc_info=True)
    raise RuntimeError(f"Could not create required directory: {e}") from e

# Verify template file exists
if not settings.template_file_path.is_file():
     logger.warning(f"Prompt template file specified but not found at: {settings.template_file_path}.")