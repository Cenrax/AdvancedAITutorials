"""
Basic Interaction Example
Demonstrates the simplest way to interact with Gemini using the Interactions API.
"""

from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BasicInteraction:
    """Handles basic text-based interactions with Gemini."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def simple_query(self, prompt: str) -> str:
        """
        Send a simple text prompt to Gemini.
        
        Args:
            prompt: The text prompt to send
            
        Returns:
            The model's response text
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt
        )
        
        # Extract text from outputs
        return interaction.outputs[-1].text
    
    def query_with_metadata(self, prompt: str) -> dict:
        """
        Send a query and return full interaction metadata.
        
        Args:
            prompt: The text prompt to send
            
        Returns:
            Dictionary containing response and metadata
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt
        )
        
        return {
            "id": interaction.id,
            "response": interaction.outputs[-1].text,
            "status": interaction.status,
            "usage": {
                "total_tokens": interaction.usage.total_tokens
            }
        }


def main():
    """Demonstrate basic interactions."""
    print("=" * 60)
    print("Basic Interaction Examples")
    print("=" * 60)
    
    bot = BasicInteraction()
    
    # Example 1: Simple query
    print("\n1. Simple Query:")
    print("-" * 60)
    response = bot.simple_query("Tell me a short joke about programming.")
    print(f"Response: {response}")
    
    # Example 2: Query with metadata
    print("\n2. Query with Metadata:")
    print("-" * 60)
    result = bot.query_with_metadata("Explain what an API is in one sentence.")
    print(f"Interaction ID: {result['id']}")
    print(f"Response: {result['response']}")
    print(f"Status: {result['status']}")
    print(f"Token Usage: {result['usage']}")
    
    # Example 3: Different types of queries
    print("\n3. Various Query Types:")
    print("-" * 60)
    
    queries = [
        "What is 25 * 47?",
        "Name three programming languages.",
        "Write a haiku about coding."
    ]
    
    for query in queries:
        response = bot.simple_query(query)
        print(f"\nQ: {query}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
