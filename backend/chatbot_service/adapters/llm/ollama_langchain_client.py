import operator
import logging
from pathlib import Path
from typing import List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chatbot_service.application.ports.llm_port import LLMPort
from chatbot_service.core.configuration.config import settings
from chatbot_service.domain.exceptions import InfrastructureException

logger = logging.getLogger(__name__)

class OllamaLangchainClient(LLMPort):
    """
    An adapter implementation of the LLMPort using Langchain with an Ollama LLM.

    Handles loading prompt templates, initializing the LLM, and constructing
    a runnable chain for generating responses based on context and chat history.
    """
    def __init__(self):
        """
        Initializes the Ollama LLM client, loads the prompt template,
        and sets up the Langchain Expression Language (LCEL) chain.
        """
        try:
            template_file_path: Path = settings.template_file_path
            logger.info(f"Attempting to load prompt template from configured path: {template_file_path}")

            # --- Load Template from File ---
            try:
                with open(template_file_path, 'r', encoding='utf-8') as f:
                    loaded_template_string = f.read()
                logger.info(f"Successfully loaded template from: {template_file_path}")
            except FileNotFoundError:
                logger.error(f"Prompt template file NOT FOUND at configured path: {template_file_path}")
                raise InfrastructureException(f"Prompt template file not found: {template_file_path}")
            except Exception as e:
                logger.error(f"Error reading template file {template_file_path}: {e}", exc_info=True)
                raise InfrastructureException(f"Error reading template file: {e}") from e

            # --- Initialize LLM ---
            self.llm = OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            logger.info(f"Initialized OllamaLLM with model: {settings.ollama_model} at base URL: {settings.ollama_base_url}")

            # --- Create PromptTemplate using the loaded string ---
            self.prompt_template = PromptTemplate(
                input_variables=["system_message", "chat_history", "context", "question"],
                template=loaded_template_string
            )

            # --- Define the Chain using LCEL ---
            # This defines the flow of data into the prompt template and then to the LLM
            self.chain = (
                {
                    "context": operator.itemgetter("context"),
                    "question": operator.itemgetter("question"),
                    "chat_history": operator.itemgetter("chat_history"),
                    "system_message": lambda x: settings.system_message
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            logger.info("LCEL chain constructed successfully.")

        except Exception as e:
             logger.exception("Failed to initialize Ollama/Langchain client")
             raise InfrastructureException(f"Failed to initialize Ollama/Langchain client: {e}") from e

    async def generate_response(self, query: str, context: str, chat_history: str) -> str:
        """
        Generates a response using the configured LLM chain.

        Args:
            query: The user's current question.
            context: The retrieved context from documents.
            chat_history: The formatted chat history string.

        Returns:
            The generated response string.

        Raises:
            InfrastructureException: If an error occurs during generation.
        """
        try:
            # Prepare the input dictionary for the chain invocation
            chain_input = {
                "question": query,
                "context": context if context else "No additional context provided.",
                "chat_history": chat_history if chat_history else "No previous conversation history."
            }
            logger.debug(f"Invoking LLM chain with input keys: {list(chain_input.keys())}")
            response = await self.chain.ainvoke(chain_input)
            logger.debug("LLM chain invocation successful.")
            return response
        except Exception as e:
            logger.exception(f"Error generating response with Ollama/Langchain for query: {query[:50]}...")
            raise InfrastructureException(f"Error generating response with Ollama/Langchain: {e}") from e