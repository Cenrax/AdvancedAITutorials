import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from openai import OpenAI, AsyncOpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReasoningEffort(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ReasoningConfig:
    effort: str = "medium"  # Default to medium instead of auto
    model: str = "o4-mini"
    analyzer_model: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_retries: int = 3
    retry_min_wait: int = 4
    retry_max_wait: int = 10

class PromptTemplates:
    SYSTEM_PROMPT = """
    You are an AI assistant that helps determine the appropriate reasoning mode for questions.
    Based on the question, determine which OpenAI reasoning mode would be most appropriate:
    
    - "low": For simple questions that need basic reasoning.
    - "medium": For questions that require moderate reasoning capabilities.
    - "high": For questions that require deep, complex reasoning.
    
    Respond with ONLY the reasoning mode as a JSON object with the format: {"mode": "<mode>"}
    """

class ReasoningModeAnalyzer:
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from the model."""
        try:
            # Clean the response string
            response = response.strip()
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response}")
            return {"mode": "medium"}  # Default to medium on parsing error
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def determine_reasoning_mode(self, question: str) -> Dict[str, str]:
        """Determine the appropriate reasoning mode for a given question.
        
        Args:
            question (str): The user's question or prompt
            
        Returns:
            dict: The reasoning configuration to use
        """
        try:
            logger.info(f"Determining reasoning mode for question: {question[:50]}...")
            response = self.client.chat.completions.create(
                model=self.config.analyzer_model,
                messages=[
                    {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Determine the reasoning mode for this question: {question}"}
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature
            )
            
            # Extract the reasoning mode from the response
            content = response.choices[0].message.content
            parsed_response = self._parse_json_response(content)
            reasoning_mode = parsed_response.get("mode", "medium")
            
            # Ensure we only use valid reasoning modes
            valid_modes = ["low", "medium", "high"]
            if reasoning_mode not in valid_modes:
                logger.warning(f"Invalid reasoning mode '{reasoning_mode}', defaulting to 'medium'")
                reasoning_mode = "medium"
            
            logger.info(f"Using reasoning mode: {reasoning_mode}")
            
            # Convert the mode to the reasoning configuration
            return {"effort": reasoning_mode}
                
        except Exception as e:
            logger.error(f"Error determining reasoning mode: {str(e)}")
            # Default to medium if there's an error
            return {"effort": "medium"}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def determine_reasoning_mode_async(self, question: str) -> Dict[str, str]:
        """Async version to determine the appropriate reasoning mode."""
        try:
            logger.info(f"Determining reasoning mode for question: {question[:50]}...")
            response = await self.async_client.chat.completions.create(
                model=self.config.analyzer_model,
                messages=[
                    {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Determine the reasoning mode for this question: {question}"}
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature
            )
            
            # Extract the reasoning mode from the response
            content = response.choices[0].message.content
            parsed_response = self._parse_json_response(content)
            reasoning_mode = parsed_response.get("mode", "medium")
            
            # Ensure we only use valid reasoning modes
            valid_modes = ["low", "medium", "high"]
            if reasoning_mode not in valid_modes:
                logger.warning(f"Invalid reasoning mode '{reasoning_mode}', defaulting to 'medium'")
                reasoning_mode = "medium"
            
            logger.info(f"Using reasoning mode: {reasoning_mode}")
            
            # Convert the mode to the reasoning configuration
            return {"effort": reasoning_mode}
                
        except Exception as e:
            logger.error(f"Error determining reasoning mode: {str(e)}")
            # Default to medium if there's an error
            return {"effort": "medium"}

class ResponseGenerator:
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_response_with_reasoning(self, prompt: str, reasoning_config: Dict[str, str]) -> str:
        """Call OpenAI with the specified reasoning mode.
        
        Args:
            prompt (str): The user's prompt
            reasoning_config (dict): The reasoning configuration to use
            
        Returns:
            str: The model's response
        """
        try:
            # Ensure we only use valid reasoning modes
            if reasoning_config.get("effort") not in ["low", "medium", "high"]:
                logger.warning(f"Invalid reasoning mode '{reasoning_config.get('effort')}', defaulting to 'medium'")
                reasoning_config = {"effort": "medium"}
                
            logger.info(f"Getting response with reasoning mode: {reasoning_config}")
            response = self.client.responses.create(
                model=self.config.model,
                reasoning=reasoning_config,
                input=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )
            
            logger.info("Response received successfully")
            return response.output_text
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response_with_reasoning_async(self, prompt: str, reasoning_config: Dict[str, str]) -> str:
        """Async version to call OpenAI with the specified reasoning mode."""
        try:
            # Ensure we only use valid reasoning modes
            if reasoning_config.get("effort") not in ["low", "medium", "high"]:
                logger.warning(f"Invalid reasoning mode '{reasoning_config.get('effort')}', defaulting to 'medium'")
                reasoning_config = {"effort": "medium"}
                
            logger.info(f"Getting response with reasoning mode: {reasoning_config}")
            response = await self.async_client.responses.create(
                model=self.config.model,
                reasoning=reasoning_config,
                input=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )
            
            logger.info("Response received successfully")
            return response.output_text
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"

class ReasoningOrchestrator:
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.analyzer = ReasoningModeAnalyzer(self.config)
        self.generator = ResponseGenerator(self.config)
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question by determining reasoning mode and getting a response.
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Result containing reasoning mode and response
        """
        try:
            # Determine the reasoning mode
            logger.info("Determining the appropriate reasoning mode...")
            reasoning_config = self.analyzer.determine_reasoning_mode(question)
            
            # Get the response with the determined reasoning mode
            logger.info("Getting response...")
            response = self.generator.get_response_with_reasoning(question, reasoning_config)
            
            return {
                "question": question,
                "reasoning_mode": reasoning_config,
                "response": response,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "success": False,
                "error": str(e)
            }
    
    async def process_question_async(self, question: str) -> Dict[str, Any]:
        """Async version to process a question."""
        try:
            # Determine the reasoning mode
            logger.info("Determining the appropriate reasoning mode...")
            reasoning_config = await self.analyzer.determine_reasoning_mode_async(question)
            
            # Get the response with the determined reasoning mode
            logger.info("Getting response...")
            response = await self.generator.get_response_with_reasoning_async(question, reasoning_config)
            
            return {
                "question": question,
                "reasoning_mode": reasoning_config,
                "response": response,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "success": False,
                "error": str(e)
            }

def main(question=None):
    # Create rich console for beautiful output
    console = Console()
    
    # Use provided question or ask for input
    if not question:
        question = input("Enter your question: ")
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Initialize token counters
    tokens_analyzer = 0
    tokens_response = 0
    
    # Create orchestrator
    orchestrator = ReasoningOrchestrator()
    
    # Process the question and track tokens
    analyzer_start = time.time()
    reasoning_config = orchestrator.analyzer.determine_reasoning_mode(question)
    analyzer_end = time.time()
    
    # Extract token usage from the analyzer's last response if available
    try:
        tokens_analyzer = orchestrator.analyzer.client.last_response.usage.total_tokens
    except (AttributeError, TypeError):
        tokens_analyzer = "N/A"
    
    # Get the response with the determined reasoning mode
    response_start = time.time()
    response = orchestrator.generator.get_response_with_reasoning(question, reasoning_config)
    response_end = time.time()
    
    # Extract token usage from the generator's last response if available
    try:
        tokens_response = orchestrator.generator.client.last_response.usage.total_tokens
    except (AttributeError, TypeError):
        tokens_response = "N/A"
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    analyzer_time = analyzer_end - analyzer_start
    response_time = response_end - response_start
    
    # Create a beautiful report
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]Reasoning Mode Analysis Report[/bold cyan]\n[dim]Generated on {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        box=box.DOUBLE
    ))
    
    # Question panel
    console.print(Panel(
        f"[bold]Question:[/bold]\n{question}",
        title="Input",
        border_style="blue"
    ))
    
    # Stats table
    stats_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Reasoning Mode", f"{reasoning_config['effort']}")
    stats_table.add_row("Analyzer Time", f"{analyzer_time:.2f} seconds")
    stats_table.add_row("Response Time", f"{response_time:.2f} seconds")
    stats_table.add_row("Total Time", f"{total_time:.2f} seconds")
    stats_table.add_row("Analyzer Tokens", f"{tokens_analyzer}")
    stats_table.add_row("Response Tokens", f"{tokens_response}")
    
    console.print(stats_table)
    
    # Response panel
    # Use a more robust approach to display the response without rich formatting
    from rich.text import Text
    
    # Create a Text object with plain styling to avoid markup interpretation
    response_text = Text(response)
    
    console.print(Panel(
        response_text,
        title="Response",
        border_style="green",
        padding=(1, 2)
    ))
    
    return {
        "question": question,
        "reasoning_mode": reasoning_config,
        "response": response,
        "stats": {
            "analyzer_time": analyzer_time,
            "response_time": response_time,
            "total_time": total_time,
            "analyzer_tokens": tokens_analyzer,
            "response_tokens": tokens_response
        }
    }

async def async_main():
    # Example usage with async
    question = input("Enter your question: ")
    
    orchestrator = ReasoningOrchestrator()
    result = await orchestrator.process_question_async(question)
    
    if result["success"]:
        print(f"\nUsing reasoning mode: {result['reasoning_mode']}")
        print("\nResponse:")
        print(result["response"])
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

def run_multiple_questions():
    # Define questions of varying difficulty
    questions = [
        # Simple question (likely low reasoning)
        "What is the capital of India?",
        
        # Medium difficulty question (likely medium reasoning)
        "Write a bash script that takes a matrix represented as a string with format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.",
        
        # Hard question (likely high reasoning)
        "Explain how quantum computing could potentially break current encryption methods like RSA, and what post-quantum cryptography solutions are being developed to address this threat. Include specific algorithms and their mathematical foundations."
    ]
    
    # Run each question and collect results
    results = []
    console = Console()
    
    for i, question in enumerate(questions):
        console.print(f"\n[bold magenta]Running Question {i+1} of {len(questions)}[/bold magenta]")
        console.print("[dim]" + "-" * 80 + "[/dim]\n")
        
        # Process the question
        result = main(question)
        results.append(result)
        
        # Add a separator between questions
        if i < len(questions) - 1:
            console.print("\n[dim]" + "=" * 80 + "[/dim]\n")
    
    # Print summary of all questions
    console.print("\n[bold cyan]Summary of All Questions[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    summary_table.add_column("Question", style="cyan")
    summary_table.add_column("Reasoning Mode", style="green")
    summary_table.add_column("Analyzer Time", style="yellow")
    summary_table.add_column("Response Time", style="yellow")
    summary_table.add_column("Total Time", style="yellow")
    
    for result in results:
        summary_table.add_row(
            result["question"][:30] + "...",
            result["reasoning_mode"]["effort"],
            f"{result['stats']['analyzer_time']:.2f}s",
            f"{result['stats']['response_time']:.2f}s",
            f"{result['stats']['total_time']:.2f}s"
        )
    
    console.print(summary_table)
    
    return results

if __name__ == "__main__":
    # Run all predefined questions
    run_multiple_questions()
    
    # For async version:
    # asyncio.run(async_main())
