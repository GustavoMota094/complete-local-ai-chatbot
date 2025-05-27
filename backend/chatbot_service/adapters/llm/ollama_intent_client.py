import operator
import logging
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chatbot_service.application.ports.intent_port import IntentPort
from chatbot_service.core.configuration.config import settings
from chatbot_service.domain.exceptions import InfrastructureException

logger = logging.getLogger(__name__)

class OllamaIntentClient(IntentPort):
    """
    An adapter implementation of the IntentPort using Langchain with an Ollama LLM
    for intent classification.

    Handles loading the intent prompt template (which is expected to contain
    the system message and the list of categories internally), initializing the LLM,
    and constructing a runnable chain for classifying user queries.
    """
    def __init__(self):
        """
        Initializes the Ollama LLM client for intent classification,
        loads the prompt template, and sets up the Langchain Expression Language (LCEL) chain.
        The prompt template file itself is expected to define the categories.
        """
        try:
            template_file_path: Path = settings.intent_template_file_path
            logger.info(f"Attempting to load intent prompt template from configured path: {template_file_path}")

            # --- Load Template from File ---
            try:
                with open(template_file_path, 'r', encoding='utf-8') as f:
                    loaded_template_string = f.read()
                logger.info(f"Successfully loaded intent template from: {template_file_path}")
            except FileNotFoundError:
                logger.error(f"Intent prompt template file NOT FOUND at configured path: {template_file_path}")
                raise InfrastructureException(f"Intent prompt template file not found: {template_file_path}")
            except Exception as e:
                logger.error(f"Error reading intent template file {template_file_path}: {e}", exc_info=True)
                raise InfrastructureException(f"Error reading intent template file: {e}") from e

            # --- Initialize LLM ---
            self.llm = OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_intent_model
            )
            logger.info(f"Initialized OllamaLLM for Intent Classification with model: {settings.ollama_intent_model} at base URL: {settings.ollama_base_url}")

            # --- Create PromptTemplate using the loaded string ---
            # Based on your provided code, input_variables is just ["question"].
            # This implies your intent_prompt_template.txt handles system message and categories internally.
            self.prompt_template = PromptTemplate(
                input_variables=["question"],
                template=loaded_template_string
            )

            # --- Define the Chain using LCEL ---
            # This defines the flow of data into the prompt template and then to the LLM.
            # Only "question" is passed to the prompt template.
            self.chain = (
                {"question": operator.itemgetter("question")}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            logger.info("LCEL chain for intent classification constructed successfully.")

            # Removed: self._valid_intent_categories and its loading,
            # as categories are now internal to the prompt file and not validated in Python.

        except Exception as e:
            logger.exception("Failed to initialize OllamaIntentClient")
            raise InfrastructureException(f"Failed to initialize OllamaIntentClient: {e}") from e

    async def classify_intent(self, query: str) -> str:
        """
        Classifies the user's query using the configured LLM chain.

        Args:
            query: The user's current question.

        Returns:
            The classified intent string as returned by the LLM (stripped of whitespace).
            Returns a default intent if an error occurs during LLM invocation.

        Raises:
            InfrastructureException: If a critical error occurs during LLM invocation that isn't handled by fallback.
        """
        try:
            # Prepare the input dictionary for the chain invocation
            chain_input = {"question": query}
            logger.debug(f"Invoking intent classification chain with input: {chain_input}")

            raw_response = await self.chain.ainvoke(chain_input)
            logger.debug(f"Raw intent classification response: '{raw_response}'")

            # Basic cleaning: strip whitespace
            classified_intent = raw_response.strip()

            if not classified_intent:
                logger.warning(
                    f"LLM returned an empty intent for query: '{query[:70]}...'. "
                    f"Defaulting to '{settings.default_intent}'."
                )
                return settings.default_intent

            logger.info(f"Classified intent for query '{query[:50]}...' as '{classified_intent}'")
            return classified_intent

        except Exception as e:
            logger.exception(f"Error during intent classification for query: {query[:50]}...")
            logger.warning(f"Returning default intent '{settings.default_intent}' due to error in classification.")
            return settings.default_intent