from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
from langchain.schema import Document

class DocumentLoaderPort(ABC):
    """
    Port defining the interface for loading documents.
    """
    @abstractmethod
    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads documents from the specified source(s).

        Args:
            source: The source to load documents from. This could be a file path (str or Path),
                    a directory path (str or Path), or a list of file/directory paths.

        Returns:
            A list of Langchain Document objects loaded from the source.
            Returns an empty list if the source is invalid or contains no loadable documents.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Subclass implementations might raise specific exceptions
                       related to file access, parsing errors, etc. (e.g., DocumentLoadingError).
        """
        pass