import os
import glob
import logging
import yaml
from typing import List, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from chatbot_service.application.ports.document_loader_port import DocumentLoaderPort
from chatbot_service.core.configuration.config import settings

# Assuming text_splitter is in the same directory or correctly pathed
from .text_splitter import split_documents

logger = logging.getLogger(__name__)

class MarkdownDocumentLoader(DocumentLoaderPort):
    """
    Loads Markdown (.md) documents from specified files or directories.

    It parses and extracts YAML front matter, cleans the content, and then uses
    a Markdown-specific splitter (via `split_documents`) to chunk the documents
    based on headers.
    """
    def __init__(self, path_to_process: str | Path | List[str] | None = None):
        """
        Initializes the loader, setting a default path if none is provided.

        Args:
            path_to_process: Optional path or list of paths to process by default.
                             Defaults to `settings.md_documents_path` if None.
        """
        self.default_path = Path(path_to_process) if path_to_process else Path(settings.md_documents_path)
        logger.info(f"MarkdownDocumentLoader initialized. Default path set to: {self.default_path}")

    def _parse_and_clean(self, raw_content: str) -> Tuple[dict, str]:
        """
        Parses YAML front matter from the beginning of the raw content.

        Args:
            raw_content: The raw string content of a Markdown file.

        Returns:
            A tuple containing:
            - A dictionary with the parsed YAML metadata (empty if no valid front matter).
            - The remaining content string after removing the front matter block.
        """
        metadata = {}
        content = raw_content
        try:
            # Check if the content starts with '---' and has a closing '---'
            if raw_content.startswith('---'):
                end_yaml_marker = '\n---\n'
                # Find closing '---' after the first one
                end_yaml_index = raw_content.find(end_yaml_marker, 3)

                if end_yaml_index != -1:
                    # Extract YAML text (without '---')
                    yaml_block = raw_content[3:end_yaml_index]
                    # Parse YAML safely
                    parsed_yaml = yaml.safe_load(yaml_block)
                    if isinstance(parsed_yaml, dict):
                        metadata = parsed_yaml # Store parsed YAML as metadata
                    else:
                        logger.warning("Parsed YAML front matter is not a dictionary. Ignoring.")

                    # Get content *after* the closing '---' marker
                    content = raw_content[end_yaml_index + len(end_yaml_marker):].lstrip()
                    logger.debug("Successfully parsed YAML front matter.")
                else:
                    # Starts with '---' but no clear end marker found, treat as content
                    logger.warning("File starts with '---' but no closing '---' found. Treating as plain content.")
                    # content remains raw_content
            # Else: No leading '---', assume no front matter or it's part of content

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML front matter: {e}", exc_info=True)
            # content remains raw_content
        except Exception as e:
             logger.error(f"Unexpected error during YAML parsing/cleaning: {e}", exc_info=True)
             # content remains raw_content

        return metadata, content

    def load(self, source: str | Path | List[str | Path]) -> List[Document]:
        """
        Loads Markdown documents from the given source(s).

        Handles individual files, directories (recursively finding .md files),
        parses YAML front matter, and prepares documents for splitting.

        Args:
            source: A single file path, directory path, or a list of paths.

        Returns:
            A list of Langchain Document objects, split according to Markdown headers.
            Returns an empty list if no valid .md files are found or processed.
        """
        all_clean_full_docs: List[Document] = [] # Store clean documents before splitting
        paths_to_load: List[Path] = []

        # Normalize input source to a list of Paths
        sources_list: List[str | Path] = source if isinstance(source, list) else [source]
        source_paths: List[Path] = [Path(s) for s in sources_list]

        # --- Find all .md files ---
        for path_item in source_paths:
            resolved_path = path_item.resolve() # Resolve path for clarity in logs
            if resolved_path.is_file() and resolved_path.suffix.lower() == '.md':
                paths_to_load.append(resolved_path)
                logger.debug(f"Identified single Markdown file: {resolved_path}")
            elif resolved_path.is_dir():
                # Use rglob for recursive search within the directory
                found_files = list(resolved_path.rglob('*.md'))
                paths_to_load.extend(found_files)
                logger.info(f"Found {len(found_files)} .md files in directory: {resolved_path}")
            else:
                logger.warning(f"Source item '{resolved_path}' is neither a valid .md file nor a directory. Skipping.")

        if not paths_to_load:
            logger.warning(f"No valid .md files found for source(s): {source}. Returning empty list.")
            return []

        # --- Load raw content, parse YAML, clean content ---
        logger.info(f"Loading raw content & processing YAML for {len(paths_to_load)} Markdown files...")
        for file_path in paths_to_load:
            try:
                # Use TextLoader for simple file reading
                loader = TextLoader(str(file_path), encoding='utf-8')
                # TextLoader returns a list; we expect one document per file
                raw_doc = loader.load()[0]

                # Extract YAML metadata and get the cleaned content
                file_metadata, clean_content = self._parse_and_clean(raw_doc.page_content)

                # Add the file path as 'source' metadata (override if YAML had 'source')
                file_metadata['source'] = str(file_path) # Use resolved path string

                # Create the final Document object with clean content and combined metadata
                clean_doc = Document(page_content=clean_content, metadata=file_metadata)
                all_clean_full_docs.append(clean_doc)
                logger.debug(f"Successfully processed YAML and content for {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to load/process file: {file_path}. Error: {e}", exc_info=True)

        if not all_clean_full_docs:
             logger.warning("No documents were processed successfully to proceed with splitting.")
             return []

        # --- Split the clean documents using the text_splitter function ---
        logger.info(f"Splitting {len(all_clean_full_docs)} clean documents using MarkdownHeaderTextSplitter...")
        try:
            # Delegate splitting to the dedicated function
            final_chunks = split_documents(all_clean_full_docs)
            logger.info(f"Total chunks created after splitting: {len(final_chunks)}")
            return final_chunks
        except Exception as e:
            # Catch potential errors during the splitting process
            logger.error(f"Failed to split documents using header splitter. Error: {e}", exc_info=True)
            return [] # Return empty list on splitting failure