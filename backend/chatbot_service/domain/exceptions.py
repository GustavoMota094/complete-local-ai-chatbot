class DomainException(Exception):
    """Base exception for errors related to domain logic or business rules."""
    pass

class ApplicationException(Exception):
    """Base exception for errors originating from the application service layer."""
    pass

class InfrastructureException(Exception):
    """Base exception for errors related to external services or infrastructure (DB, LLM API, etc.)."""
    pass

class IndexNotReadyError(InfrastructureException):
    """
    Exception raised when an operation requires the vector index, but it's not ready.
    (e.g., not loaded, not initialized, connection failed).
    """
    def __init__(self, message="Vector index not found or could not be loaded. Please index documents first or check vector store status."):
        self.message = message
        super().__init__(self.message)

class DocumentLoadingError(InfrastructureException):
    """Exception raised when loading or parsing source documents fails."""
    def __init__(self, message="Error loading or processing documents"):
        self.message = message
        super().__init__(self.message)

# Add more specific exceptions as needed, inheriting from the appropriate base class.
# Example:
# class LLMConnectionError(InfrastructureException):
#     """Raised when connection to the LLM service fails."""
#     pass