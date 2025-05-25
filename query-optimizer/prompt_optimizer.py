from openai import OpenAI
import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel
import json
from config import Config

class OptimizedPromptResponse(BaseModel):
    optimized_prompt: str
    reasoning: str = ""

class PromptOptimizer:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def build_few_shot_prompt(self, user_query: str, df_examples: pd.DataFrame) -> str:
        """Build few-shot prompt using similar examples"""
        
        few_shot_prompt = (
            "You are an expert prompt optimizer. Given a user query and examples of "
            "similar queries with user feedback, create an optimized prompt that will "
            "generate better responses based on the patterns you observe.\n\n"
            "Analyze the examples where users gave positive feedback (👍) vs negative feedback (👎) "
            "and create a prompt that incorporates the successful patterns.\n\n"
        )
        
        few_shot_prompt += "Examples of user queries, responses, and feedback:\n\n"
        
        # Add few-shot examples
        for i, row in df_examples.iterrows():
            few_shot_prompt += f"User Query: {row['conv_A_user']}\n"
            few_shot_prompt += f"Response: {row['conv_A_assistant']}\n"
            few_shot_prompt += f"User Feedback: {'👍 (Liked)' if row['conv_A_rating'] == 1 else '👎 (Disliked)'}\n"
            few_shot_prompt += f"Similarity Score: {row['similarity']:.3f}\n\n"
        
        # Add the current query
        few_shot_prompt += (
            "Based on the patterns above, create an optimized prompt that will generate "
            "a better response for the following user query. Focus on the characteristics "
            "that led to positive feedback in the examples.\n\n"
            f"Target User Query: {user_query}\n\n"
            "Please provide:\n"
            "1. An optimized prompt that incorporates successful patterns\n"
            "2. Brief reasoning for your optimization choices\n\n"
            "Format your response as JSON with 'optimized_prompt' and 'reasoning' fields."
        )
        
        return few_shot_prompt
    
    def get_optimized_prompt(self, user_query: str, similar_examples: pd.DataFrame) -> Dict[str, str]:
        """Generate optimized prompt using few-shot examples"""
        
        # Build few-shot prompt
        few_shot_prompt = self.build_few_shot_prompt(user_query, similar_examples)
        
        try:
            # Call OpenAI to generate optimized prompt
            response = self.client.responses.create(
                model=self.config.RESPONSE_MODEL,
                input=few_shot_prompt
            )
            
            # Parse the JSON response
            response_text = response.output_text.strip()
            
            # Try to extract JSON from the response
            if response_text.startswith('{') and response_text.endswith('}'):
                parsed_response = json.loads(response_text)
                return {
                    'optimized_prompt': parsed_response.get('optimized_prompt', ''),
                    'reasoning': parsed_response.get('reasoning', '')
                }
            else:
                # Fallback: treat entire response as optimized prompt
                return {
                    'optimized_prompt': response_text,
                    'reasoning': 'Generated from few-shot examples'
                }
        
        except Exception as e:
            print(f"Error generating optimized prompt: {e}")
            # Fallback to a basic prompt
            return {
                'optimized_prompt': f"Please provide a helpful and accurate response to: {user_query}",
                'reasoning': 'Fallback prompt due to generation error'
            }