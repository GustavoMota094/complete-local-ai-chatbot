import chromadb
import logging
import uuid
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from typing import List, Dict, Any

from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.core.configuration.config import settings
from chatbot_service.domain.exceptions import IndexNotReadyError, InfrastructureException

logger = logging.getLogger(__name__)

class ChromaDBStore(VectorStorePort):
    """
    Adapter implementing VectorStorePort using ChromaDB for persistent local storage.

    Manages connection to ChromaDB, handles document addition with metadata cleaning,
    and performs similarity searches using configured SentenceTransformer embeddings.
    """
    def __init__(self):
        """
        Initializes the ChromaDB client, obtains or creates the collection,
        and configures the embedding function based on application settings.
        """
        logger.info(f"Initializing ChromaDBStore...")
        logger.info(f"Persistence directory: {settings.chroma_persist_directory}")
        logger.info(f"Collection name: {settings.chroma_collection_name}")
        logger.info(f"Embedding model: {settings.embeddings_model_name}")

        self._collection = None # Initialize collection as None

        try:
            # Initialize persistent client
            self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
            logger.debug("ChromaDB persistent client initialized.")

            # Initialize the embedding function using the configured model
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embeddings_model_name
            )
            logger.debug(f"SentenceTransformer embedding function initialized with model: {settings.embeddings_model_name}")

            # Get or create the collection, specifying the embedding function and metadata (like distance metric)
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"} # Specify cosine distance for similarity search
            )
            logger.info(f"ChromaDB collection '{self._collection.name}' loaded/created successfully.")

            # Log the initial state
            initial_count = self.get_collection_count() # Use method to check readiness
            logger.info(f"Initial item count in collection '{self._collection.name}': {initial_count}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            # Ensure collection remains None if init fails
            self._collection = None
            # Raise a specific exception to indicate infrastructure failure
            raise InfrastructureException(f"Failed to initialize ChromaDB: {e}") from e

    def _clean_metadata_value(self, value: Any) -> str | int | float | bool:
        """
        Converts a single metadata value to a type compatible with ChromaDB.

        ChromaDB metadata values must be str, int, float, or bool.
        Lists or other types are converted to strings. None becomes an empty string.
        """
        if isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, list):
            # Convert list to a comma-separated string representation
            cleaned_list = [self._clean_metadata_value(item) for item in value] # Recursively clean list items
            return ", ".join(map(str, cleaned_list))
        elif value is None:
             return "" # Represent None as an empty string for compatibility
        else:
            # Convert any other type to its string representation as a fallback
            logger.debug(f"Converting metadata value of type {type(value)} to string: {str(value)[:100]}...")
            return str(value)

    def _clean_metadata_dict(self, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        """
        Cleans all key-value pairs within a metadata dictionary for ChromaDB compatibility.

        Ensures keys are strings and values are cleaned using `_clean_metadata_value`.
        """
        cleaned_meta: Dict[str, str | int | float | bool] = {}
        if metadata is None:
            return cleaned_meta # Return empty dict if input metadata is None

        for key, value in metadata.items():
            cleaned_key = str(key) # Ensure key is a string
            cleaned_value = self._clean_metadata_value(value) # Clean the value
            cleaned_meta[cleaned_key] = cleaned_value
        return cleaned_meta

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of Langchain Documents to the ChromaDB collection.

        Cleans metadata, generates unique IDs, and handles the addition process.

        Args:
            documents: A list of Document objects to add.

        Raises:
            IndexNotReadyError: If the ChromaDB collection is not initialized.
            InfrastructureException: If an error occurs during the add operation.
        """
        if not self.is_ready():
            logger.error("ChromaDBStore is not ready. Cannot add documents.")
            raise IndexNotReadyError("Vector store is not initialized properly.")

        if not documents:
             logger.warning("add_documents called with an empty list of documents. Nothing to add.")
             return # No action needed for empty list

        logger.info(f"Preparing to add {len(documents)} documents (chunks) to collection '{self._collection.name}'...")

        contents = [doc.page_content for doc in documents]
        # Clean metadata for each document *before* adding
        cleaned_metadatas = [self._clean_metadata_dict(doc.metadata) for doc in documents]
        ids = self._generate_unique_ids(documents, cleaned_metadatas)

        try:
            # Log a sample of cleaned metadata for debugging if needed
            # if cleaned_metadatas:
            #    logger.debug(f"Sample cleaned metadata [0]: {cleaned_metadatas[0]}")

            self._collection.add(
                documents=contents,
                metadatas=cleaned_metadatas, # Use the cleaned metadata
                ids=ids
            )
            logger.info(f"{len(documents)} documents added successfully to collection '{self._collection.name}'.")

            # Log count *after* successful add
            new_count = self.get_collection_count()
            logger.info(f"New item count in collection: {new_count}")

        except Exception as e:
            logger.error(f"Error during ChromaDB collection.add operation: {e}", exc_info=True)
            # Raise an exception to signal the failure
            raise InfrastructureException(f"Failed to add documents to ChromaDB: {e}") from e

    def _generate_unique_ids(self, documents: List[Document], cleaned_metadatas: List[Dict[str, Any]]) -> List[str]:
        """Helper method to generate unique IDs for documents."""
        ids = []
        used_ids_in_batch = set() # Track IDs within this batch to prevent duplicates *within the batch*

        for i, doc in enumerate(documents):
            # Prefer 'source' from cleaned metadata if available, fallback to original or placeholder
            source = cleaned_metadatas[i].get('source', doc.metadata.get('source', 'unknown_source')) if doc.metadata else 'unknown_source'
            # Ensure source component is a string
            source_str = str(source)

            # Create a base ID using source and index
            # Consider hashing content or using UUIDs for more robust uniqueness across batches if needed
            base_id = f"{source_str}_chunk_{i}"
            unique_id = base_id

            # Handle potential collisions within this batch (simple counter approach)
            counter = 0
            while unique_id in used_ids_in_batch:
                 counter += 1
                 unique_id = f"{base_id}_{counter}"
                 logger.warning(f"Generated duplicate ID '{base_id}' within batch, appending counter: '{unique_id}'")

            ids.append(unique_id)
            used_ids_in_batch.add(unique_id)

        if len(ids) != len(documents):
             # This should ideally not happen if the loop completes correctly
             raise InfrastructureException(f"ID generation resulted in {len(ids)} IDs for {len(documents)} documents.")

        logger.debug(f"Generated {len(ids)} unique IDs for batch add.")
        return ids


    def search(self, query: str, k: int) -> List[Document]:
        """
        Performs a similarity search in the ChromaDB collection.

        Args:
            query: The query string to search for.
            k: The number of top similar documents to retrieve.

        Returns:
            A list of Langchain Document objects representing the search results,
            sorted by similarity (closest first). Includes distance in metadata.
            Returns an empty list if the store is not ready, the query is empty,
            k is non-positive, or an error occurs.
        """
        if not self.is_ready():
            logger.error("ChromaDBStore is not ready. Cannot perform search.")
            # Optionally raise IndexNotReadyError instead of returning empty list
            # raise IndexNotReadyError("Vector store is not initialized properly for search.")
            return []

        if not query:
            logger.warning("Similarity search called with an empty query. Returning empty list.")
            return []
        if k <= 0:
            logger.warning(f"Similarity search called with non-positive k={k}. Returning empty list.")
            return []

        try:
            logger.info(f"Performing similarity search in collection '{self._collection.name}' for query: '{query[:50]}...' with k={k}")
            # Request documents, metadata, and distances for scoring/filtering
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # Check if results structure is as expected
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.info("No similar documents found or results format unexpected.")
                return []

            # Safely extract results, assuming the structure based on 'include'
            # Each is a list containing one list (for the single query)
            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            retrieved_contents = results.get('documents', [[]])[0]

            # Consistency check: Ensure all result lists have the same length
            if not (len(ids) == len(distances) == len(metadatas) == len(retrieved_contents)):
                 logger.error(f"Mismatch in lengths of results arrays from ChromaDB query for query '{query[:50]}...'. "
                              f"IDs: {len(ids)}, Distances: {len(distances)}, Metadatas: {len(metadatas)}, Contents: {len(retrieved_contents)}. Skipping results.")
                 return []

            matched_docs: List[Document] = []
            for i, doc_id in enumerate(ids):
                # Ensure metadata is a dict, even if None was somehow returned
                doc_metadata = metadatas[i] if isinstance(metadatas[i], dict) else {}
                # Add the distance to the metadata for potential use in scoring/ranking
                doc_metadata['distance'] = distances[i]

                # Reconstruct Langchain Document object
                doc = Document(
                    page_content=retrieved_contents[i],
                    metadata=doc_metadata # Use the retrieved and augmented metadata
                )
                matched_docs.append(doc)

            logger.info(f"Found {len(matched_docs)} similar documents for query '{query[:50]}...'.")
            return matched_docs

        except Exception as e:
            logger.error(f"Error during similarity search in ChromaDB collection '{self._collection.name}' for query '{query[:50]}...': {e}", exc_info=True)
            return [] # Return empty list on error

    def is_ready(self) -> bool:
        """
        Checks if the ChromaDB collection has been successfully initialized.

        Returns:
            True if the collection is initialized, False otherwise.
        """
        ready = self._collection is not None
        if not ready:
            logger.warning("ChromaDBStore is_ready check failed: _collection is None.")
        return ready

    def get_collection_count(self) -> int:
        """
        Gets the total number of items (documents/chunks) in the collection.

        Returns:
            The number of items in the collection.

        Raises:
            IndexNotReadyError: If the collection is not initialized.
            InfrastructureException: If an error occurs while getting the count.
        """
        if not self.is_ready():
            logger.error("Cannot get count, ChromaDBStore is not ready.")
            raise IndexNotReadyError("Vector store is not initialized properly to get count.")
        try:
            count = self._collection.count()
            logger.debug(f"Retrieved count for collection '{self._collection.name}': {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count for '{self._collection.name}': {e}", exc_info=True)
            raise InfrastructureException(f"Failed to get count from ChromaDB collection: {e}") from e