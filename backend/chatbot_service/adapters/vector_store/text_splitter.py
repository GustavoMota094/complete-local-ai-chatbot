from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.schema import Document # Use schema for Document type
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def split_documents(
    docs: List[Document],
    headers_to_split_on: Optional[List[Tuple[str, str]]] = None
) -> List[Document]:
    """
    Splits a list of Langchain Documents based on Markdown headers.

    Uses MarkdownHeaderTextSplitter to divide document content. Metadata from the
    original document and header information are merged into the resulting chunks.

    Args:
        docs: A list of Document objects to be split.
        headers_to_split_on: Optional list of (header_marker, header_name) tuples
                             to define splitting points. Defaults to H1, H2, H3.

    Returns:
        A list of Document objects representing the smaller chunks.
    """
    if headers_to_split_on is None:
        # Default headers if none are provided
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        logger.debug(f"Using default headers for splitting: {headers_to_split_on}")

    # Initialize the splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep headers in the content, useful for context
    )

    all_split_docs: List[Document] = []

    logger.info(f"Starting split process for {len(docs)} documents.")
    for i, doc in enumerate(docs):
        source_info = doc.metadata.get('source', f'unknown_source_{i}')
        if not isinstance(doc.page_content, str):
            logger.warning(f"Skipping document with non-string page_content: {source_info}")
            continue

        # Content already cleaned (YAML removed) in the loader stage
        content_to_split = doc.page_content

        # Log what's being split (useful for debugging)
        logger.debug(f"Splitting content from: {source_info}")
        # logger.debug(f"Metadata received: {doc.metadata}") # Can be verbose
        # logger.debug(f"Content being split (first 500):\n{content_to_split[:500]}\n---") # Can be verbose

        try:
            # Perform the split based on Markdown headers
            split_chunks = markdown_splitter.split_text(content_to_split)
            logger.debug(f"Document {source_info} split into {len(split_chunks)} chunks.")

            # Merge original metadata with metadata from the splitter (header info)
            for chunk in split_chunks:
                # Start with a copy of the original doc's metadata
                merged_metadata = doc.metadata.copy()
                # Update with the chunk-specific metadata (e.g., header names)
                merged_metadata.update(chunk.metadata)
                # Assign the merged metadata back to the chunk
                chunk.metadata = merged_metadata

            all_split_docs.extend(split_chunks)

        except Exception as e:
            logger.error(f"Error splitting document {source_info}: {e}", exc_info=True)
            # Decide whether to skip this doc or halt; skipping is safer
            continue

    logger.info(f"Successfully split {len(docs)} documents into {len(all_split_docs)} chunks using Markdown headers.")
    return all_split_docs