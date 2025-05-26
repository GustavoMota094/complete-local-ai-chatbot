from typing import List
from chatbot_service.application.ports.document_loader_port import DocumentLoaderPort
from chatbot_service.application.ports.vector_store_port import VectorStorePort
from chatbot_service.domain.exceptions import ApplicationException

class IndexingService:
    def __init__(self, doc_loader: DocumentLoaderPort, vector_store: VectorStorePort):
        self.doc_loader = doc_loader
        self.vector_store = vector_store

    def index_source(self, source: str | List[str]) -> str:
        try:
            print(f"Starting indexing for source {source}")
            documents = self.doc_loader.load(source)
            if not documents:
                 return f"No valid documents found or loaded from '{source}' Indexing not performed"

            print(f"{len(documents)} document(s) loaded")

            self.vector_store.add_documents(documents)

            success_message = f"Source '{source}' indexed successfully"
            print(success_message)
            return success_message

        except Exception as e:
            print(f"Error during indexing for source '{source}' {e}")
            raise ApplicationException(f"Failed to index source '{source}'") from e