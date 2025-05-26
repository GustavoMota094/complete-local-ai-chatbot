from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class VectorStorePort(ABC):
    """
    Port defining the interface for a vector store.
    """
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds a list of documents (potentially already embedded) to the vector store.

        Args:
            documents: A list of Langchain Document objects to add.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Subclass implementations might raise specific exceptions
                       related to database connections, indexing errors, etc.
                       (e.g., InfrastructureException, IndexNotReadyError).
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]:
        """
        Performs a similarity search against the documents in the vector store.

        Args:
            query: The text query to search for.
            k: The number of most similar documents to return.

        Returns:
            A list of Langchain Document objects representing the k most similar documents
            found in the store, often including similarity scores or distances in their metadata.
            Returns an empty list if no documents are found or if an error occurs.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Subclass implementations might raise specific exceptions
                       related to search execution or connection issues
                       (e.g., InfrastructureException, IndexNotReadyError).
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Checks if the vector store is properly initialized and ready for operations.

        Returns:
            True if the store is ready, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def get_collection_count(self) -> int:
        """
        Returns the total number of documents/vectors stored in the collection.

        Returns:
            The count of items in the vector store.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Subclass implementations might raise specific exceptions
                       if the count cannot be retrieved (e.g., IndexNotReadyError).
        """
        pass