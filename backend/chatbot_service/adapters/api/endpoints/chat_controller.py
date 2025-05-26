import logging
import redis
from fastapi import APIRouter, Depends, HTTPException, Response, status
from dependency_injector.wiring import inject, Provide
from chatbot_service.adapters.api.schemas import (
    IndexRequest, IndexDirectoryRequest, IndexResponse,
    ChatQuery, ChatResponse
)
from chatbot_service.application.services.indexing_service import IndexingService
from chatbot_service.application.services.chat_service import ChatService
from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.domain.exceptions import ApplicationException, IndexNotReadyError
from chatbot_service.core.dependency_injection import Container

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Chat & Indexing"]
)

# --- Indexing Endpoints ---
@router.post("/index/files",
             response_model=IndexResponse,
             summary="Index specific PDF files")
@inject
async def index_files(
    request: IndexRequest,
    indexing_service: IndexingService = Depends(Provide[Container.application.indexing_service]),
):
    """
    API endpoint to trigger indexing of a list of specified PDF file paths.
    """
    try:
        logger.info(f"Received request to index files: {request.filepaths}")
        message = indexing_service.index_source(request.filepaths)
        logger.info(f"File indexing completed: {message}")
        return IndexResponse(message=message)
    except ApplicationException as e:
        logger.error(f"Application error during file indexing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during file indexing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing file indexing")

@router.post("/index/directory",
             response_model=IndexResponse,
             summary="Index all PDF files in a directory")
@inject
async def index_directory(
    request: IndexDirectoryRequest,
    indexing_service: IndexingService = Depends(Provide[Container.application.indexing_service]),
):
    """
    API endpoint to trigger indexing of all PDF files within a specified directory.
    """
    try:
        logger.info(f"Received request to index directory: {request.directory_path}")
        message = indexing_service.index_source(request.directory_path)
        logger.info(f"Directory indexing completed: {message}")
        return IndexResponse(message=message)
    except ApplicationException as e:
        logger.error(f"Application error during directory indexing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during directory indexing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing directory indexing")

# --- Chat Endpoint ---
@router.post("/chat",
             response_model=ChatResponse,
             summary="Send question to Chatbot")
@inject
async def chat(
    query: ChatQuery,
    chat_service: ChatService = Depends(Provide[Container.application.chat_service]),
):
    """
    API endpoint to receive a user query, interact with the chat service,
    and return the chatbot's response. Handles session management.
    """
    try:
        session_id = query.session_id
        logger.info(f"Processing chat request for session ID: {session_id}")

        response_text = await chat_service.generate_chat_response(
            query=query.query,
            session_id=session_id
        )
        logger.info(f"Chat response generated successfully for session ID: {session_id}")
        return ChatResponse(response=response_text, session_id=session_id)

    except IndexNotReadyError as e:
         logger.warning(f"IndexNotReadyError encountered for session {query.session_id}: {e}")
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ApplicationException as e:
        logger.error(f"Application error during chat processing for session {query.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /chat for session {query.session_id if query else 'unknown'}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing chat question")

# --- Clear History Endpoint ---
@router.delete("/chat/{session_id}/history",
               status_code=status.HTTP_204_NO_CONTENT,
               summary="Clear chat history for a session")
@inject
async def clear_history(
    session_id: str,
    chat_service: ChatService = Depends(Provide[Container.application.chat_service])
):
    """
    API endpoint to clear the chat history associated with a specific session ID.
    """
    try:
        logger.info(f"Received request to clear history for session ID: {session_id}")
        chat_service.clear_chat_history(session_id)
        logger.info(f"Successfully cleared history for session ID: {session_id}")
        # No response body needed for 204
    except ApplicationException as e:
        logger.error(f"Application error while clearing history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error clearing history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error clearing chat history")