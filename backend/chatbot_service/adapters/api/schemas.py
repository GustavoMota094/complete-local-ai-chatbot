from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class IndexRequest(BaseModel):
    """Schema for requesting indexing of specific file paths."""
    filepaths: List[str]

class IndexDirectoryRequest(BaseModel):
    """Schema for requesting indexing of a directory."""
    directory_path: str

class IndexResponse(BaseModel):
    """Schema for the response after an indexing operation."""
    message: str

class ChatQuery(BaseModel):
    """Schema for sending a query to the chatbot."""
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class ChatResponse(BaseModel):
    """Schema for the chatbot's response."""
    response: str
    session_id: str