from abc import ABC, abstractmethod

class LLMPort(ABC):
    """
    Port defining the interface for Language Model interactions.
    """
    @abstractmethod
    async def generate_response(self, query: str, context: str, chat_history: str) -> str:
        """
        Generates a text response based on the provided inputs.

        Args:
            query: The user's input query or question.
            context: Relevant contextual information retrieved from a knowledge source.
            chat_history: A string representation of the conversation history.

        Returns:
            The generated text response from the language model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Subclass implementations might raise specific exceptions
                       related to API calls or model errors (e.g., InfrastructureException).
        """
        pass