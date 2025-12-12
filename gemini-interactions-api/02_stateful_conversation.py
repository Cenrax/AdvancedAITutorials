"""
Stateful Conversation Example
Demonstrates server-side state management using previous_interaction_id.
"""

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


class StatefulConversation:
    """Manages stateful conversations with automatic context tracking."""
    
    def __init__(self):
        """Initialize the client and conversation state."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.current_interaction_id = None
    
    def send_message(self, message: str) -> str:
        """
        Send a message and maintain conversation context.
        
        Args:
            message: User message to send
            
        Returns:
            Model's response text
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=message,
            previous_interaction_id=self.current_interaction_id
        )
        
        # Update conversation state
        self.current_interaction_id = interaction.id
        
        return interaction.outputs[-1].text
    
    def get_conversation_history(self) -> dict:
        """
        Retrieve the full conversation history from the server.
        
        Returns:
            The complete interaction object
        """
        if not self.current_interaction_id:
            return None
        
        return self.client.interactions.get(self.current_interaction_id)
    
    def reset_conversation(self):
        """Start a new conversation by clearing the interaction ID."""
        self.current_interaction_id = None


def main():
    """Demonstrate stateful conversations."""
    print("=" * 60)
    print("Stateful Conversation Examples")
    print("=" * 60)
    
    # Example 1: Multi-turn conversation with context
    print("\n1. Multi-turn Conversation:")
    print("-" * 60)
    
    chat = StatefulConversation()
    
    # Turn 1: Introduce yourself
    response1 = chat.send_message("Hi, my name is Alice and I'm a software engineer.")
    print(f"User: Hi, my name is Alice and I'm a software engineer.")
    print(f"Bot: {response1}\n")
    
    # Turn 2: Ask about name (tests context retention)
    response2 = chat.send_message("What's my name?")
    print(f"User: What's my name?")
    print(f"Bot: {response2}\n")
    
    # Turn 3: Ask about profession (tests context retention)
    response3 = chat.send_message("What do I do for work?")
    print(f"User: What do I do for work?")
    print(f"Bot: {response3}\n")
    
    # Turn 4: Follow-up question
    response4 = chat.send_message("Can you recommend a programming language for me?")
    print(f"User: Can you recommend a programming language for me?")
    print(f"Bot: {response4}")
    
    # Example 2: Retrieve conversation history
    print("\n2. Conversation History:")
    print("-" * 60)
    
    history = chat.get_conversation_history()
    print(f"Interaction ID: {history.id}")
    print(f"Status: {history.status}")
    print(f"Total Tokens Used: {history.usage.total_tokens}")
    
    # Example 3: Reset and start new conversation
    print("\n3. New Conversation After Reset:")
    print("-" * 60)
    
    chat.reset_conversation()
    
    response5 = chat.send_message("What's my name?")
    print(f"User: What's my name?")
    print(f"Bot: {response5}")
    print("(Notice: The bot doesn't remember Alice)")


class ConversationManager:
    """Advanced conversation manager with multiple sessions."""
    
    def __init__(self):
        """Initialize the client and session storage."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.sessions = {}
    
    def send_message(self, session_id: str, message: str) -> str:
        """
        Send a message in a specific session.
        
        Args:
            session_id: Unique identifier for the conversation session
            message: User message to send
            
        Returns:
            Model's response text
        """
        previous_id = self.sessions.get(session_id)
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=message,
            previous_interaction_id=previous_id
        )
        
        # Update session state
        self.sessions[session_id] = interaction.id
        
        return interaction.outputs[-1].text
    
    def delete_session(self, session_id: str):
        """Delete a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


def demo_multi_session():
    """Demonstrate managing multiple conversation sessions."""
    print("\n" + "=" * 60)
    print("Multi-Session Management")
    print("=" * 60)
    
    manager = ConversationManager()
    
    # Session 1: Tech support
    print("\nSession 1 (Tech Support):")
    print("-" * 60)
    response = manager.send_message("tech_support", "I'm having issues with my Python installation.")
    print(f"User: I'm having issues with my Python installation.")
    print(f"Bot: {response}\n")
    
    # Session 2: Recipe help
    print("Session 2 (Recipe Help):")
    print("-" * 60)
    response = manager.send_message("recipe_help", "How do I make chocolate chip cookies?")
    print(f"User: How do I make chocolate chip cookies?")
    print(f"Bot: {response}\n")
    
    # Continue Session 1
    print("Back to Session 1:")
    print("-" * 60)
    response = manager.send_message("tech_support", "What was my issue again?")
    print(f"User: What was my issue again?")
    print(f"Bot: {response}")
    print("(Notice: Context is maintained per session)")


if __name__ == "__main__":
    main()
    demo_multi_session()
