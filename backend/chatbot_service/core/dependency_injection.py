from dependency_injector import containers, providers
from chatbot_service.core.configuration.config import settings

# --- Ports (Interfaces) ---
from chatbot_service.application.ports.conversation_logger_port import ConversationLoggerPort
from chatbot_service.application.ports.document_loader_port import DocumentLoaderPort
from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.application.ports.llm_port import LLMPort
from chatbot_service.application.ports.intent_port import IntentPort

# --- Adapters (Implementations) ---
from chatbot_service.adapters.database.postgres_conversation_logger import PostgresConversationLogger
from chatbot_service.adapters.llm.ollama_langchain_client import OllamaLangchainClient
from chatbot_service.adapters.llm.ollama_intent_client import OllamaIntentClient
from chatbot_service.adapters.vector_store.document_loader import MarkdownDocumentLoader
from chatbot_service.adapters.vector_store.chromadb_store import ChromaDBStore

# --- Application Services ---
from chatbot_service.application.services.indexing_service import IndexingService
from chatbot_service.application.services.chat_service import ChatService

class Adapters(containers.DeclarativeContainer):
    """
    Container for infrastructure adapters (implementations of ports).
    """
    config = providers.Object(settings)
    postgres_pool = providers.Singleton(lambda: None)

    document_loader: providers.Singleton[DocumentLoaderPort] = providers.Singleton(
        MarkdownDocumentLoader
    )

    vector_store: providers.Singleton[VectorStorePort] = providers.Singleton(
        ChromaDBStore
    )

    conversation_logger: providers.Singleton[ConversationLoggerPort] = providers.Singleton(
        PostgresConversationLogger,
        pool=postgres_pool
    )

    llm_client: providers.Singleton[LLMPort] = providers.Singleton(
        OllamaLangchainClient
    )

    intent_client: providers.Singleton[IntentPort] = providers.Singleton(
        OllamaIntentClient
    )

class Application(containers.DeclarativeContainer):
    """
    Container for application services.
    """
    adapters = providers.DependenciesContainer()

    indexing_service = providers.Factory(
        IndexingService,
        doc_loader=adapters.document_loader,
        vector_store=adapters.vector_store,
    )

    chat_service = providers.Singleton(
        ChatService,
        vector_store=adapters.vector_store,
        llm_client=adapters.llm_client,
        conversation_logger=adapters.conversation_logger,
        intent_client=adapters.intent_client
    )

class Container(containers.DeclarativeContainer):
    """
    Root dependency injection container for the application.
    """
    adapters = providers.Container(Adapters)
    application = providers.Container(Application, adapters=adapters)

container = Container()