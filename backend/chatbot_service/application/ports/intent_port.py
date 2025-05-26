from abc import ABC, abstractmethod
from typing import List

class IntentPort(ABC):
    """
    Port defining the interface for classifying user intent.
    """
    @abstractmethod
    async def classify_intent(self, query: str) -> str:
        """
        Classifies the user's query into a predefined intent category.

        Args:
            query: The user's input query.

        Returns:
            A string representing the classified intent (e.g., "Password Reset").
            It might return a default/unknown intent if classification fails.
        """
        pass