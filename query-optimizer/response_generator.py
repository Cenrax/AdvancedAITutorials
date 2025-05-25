from openai import OpenAI
from typing import Dict, Any
from config import Config

class ResponseGenerator:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def generate_response_with_optimized_prompt(self, optimized_prompt: str, user_query: str) -> str:
        """Generate response using optimized prompt"""
        
        final_prompt = f"{optimized_prompt}\n\nUser Query: {user_query}\n\nResponse:"
        
        try:
            response = self.client.responses.create(
                model=self.config.RESPONSE_MODEL,
                input=final_prompt
            )
            return response.output_text.strip()
        
        except Exception as e:
            print(f"Error generating optimized response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_baseline_response(self, user_query: str) -> str:
        """Generate baseline response without optimization"""
        
        baseline_prompt = f"Please provide a helpful response to the following query:\n\nUser Query: {user_query}\n\nResponse:"
        
        try:
            response = self.client.responses.create(
                model=self.config.RESPONSE_MODEL,
                input=baseline_prompt
            )
            return response.output_text.strip()
        
        except Exception as e:
            print(f"Error generating baseline response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_both_responses(self, user_query: str, optimized_prompt: str) -> Dict[str, str]:
        """Generate both optimized and baseline responses"""
        
        return {
            'optimized_response': self.generate_response_with_optimized_prompt(optimized_prompt, user_query),
            'baseline_response': self.generate_baseline_response(user_query)
        }