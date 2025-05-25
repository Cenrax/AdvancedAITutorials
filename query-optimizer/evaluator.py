from openai import OpenAI
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel
import json
from config import Config

class JudgeResponse(BaseModel):
    score: int  # 0 or 1
    reasoning: str = ""

class Evaluator:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def build_judge_prompt(self, user_query: str, response: str, few_shot_examples: pd.DataFrame) -> str:
        """Build prompt for LLM-as-a-judge evaluation"""
        
        judge_prompt = (
            "You are an expert evaluator. Based on the examples provided, "
            "evaluate if the response satisfies the user query and would likely "
            "receive positive feedback from the user.\n\n"
            "Look at the patterns in the examples below to understand what makes "
            "a good vs bad response:\n\n"
        )
        
        # Add few-shot examples for context
        for i, row in few_shot_examples.iterrows():
            judge_prompt += f"User Query: {row['conv_A_user']}\n"
            judge_prompt += f"Response: {row['conv_A_assistant']}\n"
            judge_prompt += f"User Feedback: {'👍 (Liked)' if row['conv_A_rating'] == 1 else '👎 (Disliked)'}\n\n"
        
        # Add current evaluation
        judge_prompt += (
            "Now, evaluate the following response based on the patterns above:\n\n"
            f"User Query: {user_query}\n"
            f"Response: {response}\n\n"
            "Return a JSON with:\n"
            "- 'score': 1 if the response is good (likely to be liked), 0 if bad (likely to be disliked)\n"
            "- 'reasoning': Brief explanation of your evaluation\n\n"
            "JSON Response:"
        )
        
        return judge_prompt
    
    def judge_response(self, user_query: str, response: str, few_shot_examples: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM as judge to evaluate response quality"""
        
        judge_prompt = self.build_judge_prompt(user_query, response, few_shot_examples)
        
        try:
            # Call judge model
            judge_response = self.client.responses.create(
                model=self.config.JUDGE_MODEL,
                input=judge_prompt
            )
            
            response_text = judge_response.output_text.strip()
            
            # Try to parse JSON response
            if response_text.startswith('{') and response_text.endswith('}'):
                parsed_response = json.loads(response_text)
                score = int(parsed_response.get('score', 0))
                reasoning = parsed_response.get('reasoning', '')
            else:
                # Fallback: try to extract score from text
                if '1' in response_text and 'good' in response_text.lower():
                    score = 1
                    reasoning = "Extracted from text analysis"
                else:
                    score = 0
                    reasoning = "Extracted from text analysis"
            
            return {
                'score': max(0, min(1, score)),  # Ensure score is 0 or 1
                'reasoning': reasoning
            }
        
        except Exception as e:
            print(f"Error in judge evaluation: {e}")
            return {'score': 0, 'reasoning': f"Error in evaluation: {str(e)}"}
    
    def evaluate_responses(self, 
                         user_query: str, 
                         optimized_response: str, 
                         baseline_response: str, 
                         few_shot_examples: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate both optimized and baseline responses"""
        
        optimized_eval = self.judge_response(user_query, optimized_response, few_shot_examples)
        baseline_eval = self.judge_response(user_query, baseline_response, few_shot_examples)
        
        return {
            'optimized_score': optimized_eval['score'],
            'optimized_reasoning': optimized_eval['reasoning'],
            'baseline_score': baseline_eval['score'],
            'baseline_reasoning': baseline_eval['reasoning']
        }