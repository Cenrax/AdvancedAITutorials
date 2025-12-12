"""
Built-in Tools Example
Demonstrates using Google Search, Code Execution, and URL Context tools.
"""

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


class BuiltInToolsAgent:
    """Agent that uses Gemini's built-in tools."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def search_with_google(self, query: str) -> str:
        """
        Use Google Search grounding to answer queries with current information.
        
        Args:
            query: Search query
            
        Returns:
            Response grounded in search results
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=query,
            tools=[{"type": "google_search"}]
        )
        
        # Extract text output (not the search result metadata)
        text_output = next((o for o in interaction.outputs if o.type == "text"), None)
        if text_output:
            return text_output.text
        return "No text response generated."
    
    def execute_code(self, prompt: str) -> str:
        """
        Use code execution to solve computational problems.
        
        Args:
            prompt: Problem description
            
        Returns:
            Response with code execution results
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt,
            tools=[{"type": "code_execution"}]
        )
        
        return interaction.outputs[-1].text
    
    def analyze_url(self, url: str, question: str) -> str:
        """
        Fetch and analyze content from a URL.
        
        Args:
            url: URL to analyze
            question: Question about the URL content
            
        Returns:
            Analysis of the URL content
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=f"{question}\n\nURL: {url}",
            tools=[{"type": "url_context"}]
        )
        
        # Extract text output
        text_output = next((o for o in interaction.outputs if o.type == "text"), None)
        if text_output:
            return text_output.text
        return "No text response generated."
    
    def combine_tools(self, prompt: str, tools: list) -> str:
        """
        Use multiple built-in tools together.
        
        Args:
            prompt: User prompt
            tools: List of tool types to enable
            
        Returns:
            Response using available tools
        """
        tool_configs = [{"type": tool_type} for tool_type in tools]
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt,
            tools=tool_configs
        )
        
        # Extract text output
        text_output = next((o for o in interaction.outputs if o.type == "text"), None)
        if text_output:
            return text_output.text
        return "No text response generated."


def demo_google_search():
    """Demonstrate Google Search grounding."""
    print("=" * 60)
    print("Google Search Grounding")
    print("=" * 60)
    
    agent = BuiltInToolsAgent()
    
    # Example 1: Current events
    print("\n1. Current Events Query:")
    print("-" * 60)
    response = agent.search_with_google("Who won the last Super Bowl?")
    print(f"Query: Who won the last Super Bowl?")
    print(f"Response: {response}")
    
    # Example 2: Recent information
    print("\n2. Recent Information:")
    print("-" * 60)
    response = agent.search_with_google("What are the latest AI developments in 2024?")
    print(f"Query: What are the latest AI developments in 2024?")
    print(f"Response: {response}")
    
    # Example 3: Factual lookup
    print("\n3. Factual Lookup:")
    print("-" * 60)
    response = agent.search_with_google("What is the population of Tokyo?")
    print(f"Query: What is the population of Tokyo?")
    print(f"Response: {response}")


def demo_code_execution():
    """Demonstrate code execution capabilities."""
    print("\n" + "=" * 60)
    print("Code Execution")
    print("=" * 60)
    
    agent = BuiltInToolsAgent()
    
    # Example 1: Mathematical computation
    print("\n1. Mathematical Computation:")
    print("-" * 60)
    response = agent.execute_code("Calculate the 50th Fibonacci number.")
    print(f"Query: Calculate the 50th Fibonacci number.")
    print(f"Response: {response}")
    
    # Example 2: Data analysis
    print("\n2. Data Analysis:")
    print("-" * 60)
    response = agent.execute_code(
        "Calculate the mean, median, and standard deviation of the numbers: 12, 15, 18, 22, 25, 28, 30, 35, 40, 45"
    )
    print(f"Query: Calculate statistics for a dataset")
    print(f"Response: {response}")
    
    # Example 3: Algorithm implementation
    print("\n3. Algorithm Implementation:")
    print("-" * 60)
    response = agent.execute_code(
        "Write a function to check if a number is prime, then find all prime numbers between 1 and 50."
    )
    print(f"Query: Find prime numbers between 1 and 50")
    print(f"Response: {response}")
    
    # Example 4: Complex calculation
    print("\n4. Complex Calculation:")
    print("-" * 60)
    response = agent.execute_code(
        "Calculate compound interest for $10,000 invested at 5% annual rate for 10 years, compounded monthly."
    )
    print(f"Query: Compound interest calculation")
    print(f"Response: {response}")


def demo_url_context():
    """Demonstrate URL context fetching."""
    print("\n" + "=" * 60)
    print("URL Context")
    print("=" * 60)
    
    agent = BuiltInToolsAgent()
    
    # Example 1: Summarize webpage
    print("\n1. Summarize Webpage:")
    print("-" * 60)
    response = agent.analyze_url(
        "https://www.wikipedia.org/",
        "Summarize the main purpose of this website."
    )
    print(f"URL: https://www.wikipedia.org/")
    print(f"Response: {response}")
    
    # Example 2: Extract specific information
    print("\n2. Extract Specific Information:")
    print("-" * 60)
    response = agent.analyze_url(
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "When was Python first released and who created it?"
    )
    print(f"URL: Python Wikipedia page")
    print(f"Response: {response}")


def demo_combined_tools():
    """Demonstrate using multiple tools together."""
    print("\n" + "=" * 60)
    print("Combined Tools Usage")
    print("=" * 60)
    
    agent = BuiltInToolsAgent()
    
    # Example 1: Search + Code Execution
    print("\n1. Search + Code Execution:")
    print("-" * 60)
    response = agent.combine_tools(
        "Find the current price of Bitcoin and calculate how much 5 Bitcoins would be worth.",
        tools=["google_search", "code_execution"]
    )
    print(f"Query: Bitcoin price calculation")
    print(f"Response: {response}")
    
    # Example 2: URL Context + Code Execution
    print("\n2. URL Context + Code Execution:")
    print("-" * 60)
    response = agent.combine_tools(
        "Fetch data from https://www.wikipedia.org/ and count how many times the word 'encyclopedia' appears.",
        tools=["url_context", "code_execution"]
    )
    print(f"Query: Word count from URL")
    print(f"Response: {response}")


class ResearchAssistant:
    """Research assistant using built-in tools."""
    
    def __init__(self):
        """Initialize the client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def research_topic(self, topic: str) -> dict:
        """
        Research a topic using Google Search and provide analysis.
        
        Args:
            topic: Topic to research
            
        Returns:
            Dictionary with findings and analysis
        """
        # Step 1: Search for information
        print(f"🔍 Researching: {topic}")
        search_interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=f"Find recent information about: {topic}",
            tools=[{"type": "google_search"}]
        )
        
        # Step 2: Analyze and summarize
        print("📊 Analyzing findings...")
        analysis_interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input="Provide a structured summary with key points and insights.",
            previous_interaction_id=search_interaction.id
        )
        
        # Extract text outputs
        search_text = next((o.text for o in search_interaction.outputs if o.type == "text"), "")
        analysis_text = next((o.text for o in analysis_interaction.outputs if o.type == "text"), "")
        
        return {
            "topic": topic,
            "findings": search_text,
            "analysis": analysis_text
        }


def demo_research_assistant():
    """Demonstrate research assistant workflow."""
    print("\n" + "=" * 60)
    print("Research Assistant Workflow")
    print("=" * 60)
    
    assistant = ResearchAssistant()
    
    print("\nResearching AI Safety:")
    print("-" * 60)
    
    try:
        result = assistant.research_topic("AI Safety and Alignment")
        print(f"\n📋 Findings:\n{result['findings']}")
        print(f"\n💡 Analysis:\n{result['analysis']}")
    except Exception as e:
        print(f"Error: {e}")


class DataAnalyst:
    """Data analyst using code execution."""
    
    def __init__(self):
        """Initialize the client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def analyze_dataset(self, data_description: str, analysis_request: str) -> str:
        """
        Analyze a dataset using code execution.
        
        Args:
            data_description: Description of the dataset
            analysis_request: What analysis to perform
            
        Returns:
            Analysis results
        """
        prompt = f"""
Dataset: {data_description}

Task: {analysis_request}

Please write Python code to perform this analysis and provide insights.
        """
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt,
            tools=[{"type": "code_execution"}]
        )
        
        return interaction.outputs[-1].text


def demo_data_analyst():
    """Demonstrate data analyst workflow."""
    print("\n" + "=" * 60)
    print("Data Analyst Workflow")
    print("=" * 60)
    
    analyst = DataAnalyst()
    
    print("\nAnalyzing Sales Data:")
    print("-" * 60)
    
    result = analyst.analyze_dataset(
        data_description="Monthly sales: [15000, 18000, 22000, 19000, 25000, 28000, 30000, 27000, 32000, 35000, 38000, 42000]",
        analysis_request="Calculate growth rate, identify trends, and predict next month's sales using linear regression."
    )
    
    print(f"Analysis:\n{result}")


if __name__ == "__main__":
    demo_google_search()
    demo_code_execution()
    demo_url_context()
    demo_combined_tools()
    demo_research_assistant()
    demo_data_analyst()
    
    print("\n" + "=" * 60)
    print("Summary of Built-in Tools")
    print("=" * 60)
    print("""
1. Google Search (google_search):
   - Get current, real-time information
   - Answer questions about recent events
   - Ground responses in factual data

2. Code Execution (code_execution):
   - Perform complex calculations
   - Analyze data programmatically
   - Implement algorithms
   - Generate visualizations

3. URL Context (url_context):
   - Fetch and analyze web pages
   - Extract information from URLs
   - Summarize online content

These tools can be combined for powerful workflows!
    """)
