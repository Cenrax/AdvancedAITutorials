"""
Stateless Conversation Example
Demonstrates client-side history management without server-side state.
"""

from google import genai
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class StatelessConversation:
    """Manages conversations with client-side history tracking."""
    
    def __init__(self):
        """Initialize the client and conversation history."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.history: List[Dict] = []
    
    def send_message(self, message: str) -> str:
        """
        Send a message with full conversation history.
        
        Args:
            message: User message to send
            
        Returns:
            Model's response text
        """
        # Add user message to history
        self.history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        
        # Send entire history to model
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=self.history,
            store=False  # Opt out of server-side storage
        )
        
        # Add model response to history
        self.history.append({
            "role": "model",
            "content": interaction.outputs
        })
        
        return interaction.outputs[-1].text
    
    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
    
    def export_history(self) -> str:
        """
        Export conversation history as formatted text.
        
        Returns:
            Formatted conversation string
        """
        output = []
        for turn in self.history:
            role = turn["role"].upper()
            if turn["role"] == "user":
                text = turn["content"][0]["text"]
            else:
                text = turn["content"][-1].text if hasattr(turn["content"][-1], 'text') else str(turn["content"][-1])
            output.append(f"{role}: {text}")
        return "\n\n".join(output)


def main():
    """Demonstrate stateless conversations."""
    print("=" * 60)
    print("Stateless Conversation Examples")
    print("=" * 60)
    
    # Example 1: Basic stateless conversation
    print("\n1. Basic Stateless Conversation:")
    print("-" * 60)
    
    chat = StatelessConversation()
    
    # Turn 1
    response1 = chat.send_message("What are the three largest cities in Spain?")
    print(f"User: What are the three largest cities in Spain?")
    print(f"Bot: {response1}\n")
    
    # Turn 2: Follow-up question
    response2 = chat.send_message("What is the most famous landmark in the second one?")
    print(f"User: What is the most famous landmark in the second one?")
    print(f"Bot: {response2}\n")
    
    # Turn 3: Another follow-up
    response3 = chat.send_message("Tell me about its history.")
    print(f"User: Tell me about its history.")
    print(f"Bot: {response3}")
    
    # Example 2: View conversation history
    print("\n2. Conversation History:")
    print("-" * 60)
    print(f"Total turns: {len(chat.get_history())}")
    print("\nFormatted History:")
    print(chat.export_history())
    
    # Example 3: Clear and restart
    print("\n3. After Clearing History:")
    print("-" * 60)
    
    chat.clear_history()
    response4 = chat.send_message("What were we just talking about?")
    print(f"User: What were we just talking about?")
    print(f"Bot: {response4}")
    print("(Notice: The bot has no memory of previous conversation)")


class ContextWindowManager:
    """Manages conversation history with context window limits."""
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize with a maximum number of turns to keep.
        
        Args:
            max_turns: Maximum number of conversation turns to maintain
        """
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.history: List[Dict] = []
        self.max_turns = max_turns
    
    def send_message(self, message: str) -> str:
        """
        Send a message with limited history window.
        
        Args:
            message: User message to send
            
        Returns:
            Model's response text
        """
        # Add user message
        self.history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        
        # Trim history to max_turns (keep most recent)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        
        # Send trimmed history
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=self.history,
            store=False
        )
        
        # Add model response
        self.history.append({
            "role": "model",
            "content": interaction.outputs
        })
        
        # Trim again if needed
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        
        return interaction.outputs[-1].text


def demo_context_window():
    """Demonstrate context window management."""
    print("\n" + "=" * 60)
    print("Context Window Management")
    print("=" * 60)
    
    # Create manager with small window
    chat = ContextWindowManager(max_turns=4)
    
    print("\nSending 5 messages (window size = 4):")
    print("-" * 60)
    
    messages = [
        "My favorite color is blue.",
        "I like pizza.",
        "I work as a teacher.",
        "I have a dog named Max.",
        "What's my favorite color?"
    ]
    
    for i, msg in enumerate(messages, 1):
        response = chat.send_message(msg)
        print(f"\n{i}. User: {msg}")
        print(f"   Bot: {response}")
        print(f"   History size: {len(chat.history)} turns")
    
    print("\n(Notice: The bot may not remember the color because it was pushed out of the context window)")


class SelectiveHistoryManager:
    """Manages conversation with selective history inclusion."""
    
    def __init__(self):
        """Initialize the client and history storage."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.full_history: List[Dict] = []
        self.important_turns: List[int] = []
    
    def send_message(self, message: str, important: bool = False) -> str:
        """
        Send a message and optionally mark it as important.
        
        Args:
            message: User message to send
            important: Whether to always include this turn in context
            
        Returns:
            Model's response text
        """
        turn_index = len(self.full_history)
        
        # Add user message
        self.full_history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        
        if important:
            self.important_turns.append(turn_index)
        
        # Build context: important turns + recent turns
        context = []
        recent_start = max(0, len(self.full_history) - 6)
        
        for i, turn in enumerate(self.full_history):
            if i in self.important_turns or i >= recent_start:
                context.append(turn)
        
        # Send selective history
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=context,
            store=False
        )
        
        # Add model response
        self.full_history.append({
            "role": "model",
            "content": interaction.outputs
        })
        
        return interaction.outputs[-1].text


def demo_selective_history():
    """Demonstrate selective history management."""
    print("\n" + "=" * 60)
    print("Selective History Management")
    print("=" * 60)
    
    chat = SelectiveHistoryManager()
    
    print("\nMarking important information:")
    print("-" * 60)
    
    # Important context
    response1 = chat.send_message("My account number is 12345.", important=True)
    print(f"User: My account number is 12345. [IMPORTANT]")
    print(f"Bot: {response1}\n")
    
    # Filler conversation
    chat.send_message("How's the weather?")
    chat.send_message("Tell me a joke.")
    chat.send_message("What's 2+2?")
    
    # Reference important context
    response2 = chat.send_message("What's my account number?")
    print(f"User: What's my account number?")
    print(f"Bot: {response2}")
    print("(Notice: The bot remembers the account number despite filler messages)")


if __name__ == "__main__":
    main()
    demo_context_window()
    demo_selective_history()
