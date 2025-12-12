"""
Deep Research Agent Example
Demonstrates using the Deep Research agent for long-running research tasks.
"""

import time
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


class DeepResearchAgent:
    """Handles deep research tasks using Gemini's specialized research agent."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def research(self, topic: str, background: bool = True) -> dict:
        """
        Conduct deep research on a topic.
        
        Args:
            topic: Research topic or question
            background: Whether to run in background mode
            
        Returns:
            Dictionary with research results and metadata
        """
        print(f"Starting research on: {topic}")
        print(f"Mode: {'Background' if background else 'Foreground'}")
        print("-" * 60)
        
        # Start the Deep Research Agent
        initial_interaction = self.client.interactions.create(
            input=topic,
            agent="deep-research-pro-preview-12-2025",
            background=background
        )
        
        print(f"Research started. Interaction ID: {initial_interaction.id}")
        
        # Poll for results
        iteration = 0
        while True:
            interaction = self.client.interactions.get(initial_interaction.id)
            iteration += 1
            
            print(f"[Poll {iteration}] Status: {interaction.status}")
            
            if interaction.status == "completed":
                print("\nResearch completed successfully!")
                return {
                    "id": interaction.id,
                    "status": interaction.status,
                    "report": interaction.outputs[-1].text,
                    "usage": {
                        "total_tokens": interaction.usage.total_tokens
                    }
                }
            elif interaction.status in ["failed", "cancelled"]:
                print(f"\nResearch failed with status: {interaction.status}")
                return {
                    "id": interaction.id,
                    "status": interaction.status,
                    "error": "Research task failed or was cancelled"
                }
            
            # Wait before next poll
            time.sleep(10)
    
    def get_research_status(self, interaction_id: str) -> dict:
        """
        Check the status of a research task.
        
        Args:
            interaction_id: ID of the research interaction
            
        Returns:
            Status information
        """
        interaction = self.client.interactions.get(interaction_id)
        
        return {
            "id": interaction.id,
            "status": interaction.status,
            "is_complete": interaction.status == "completed"
        }


def main():
    """Demonstrate Deep Research Agent capabilities."""
    print("=" * 60)
    print("Deep Research Agent Examples")
    print("=" * 60)
    
    agent = DeepResearchAgent()
    
    # Example 1: Research a technical topic
    print("\n1. Technical Research:")
    print("-" * 60)
    
    result = agent.research(
        "Research the history of Google TPUs with a focus on 2025 and 2026.",
        background=True
    )
    
    if result["status"] == "completed":
        print("\nFinal Report:")
        print("=" * 60)
        print(result["report"])
        print("\n" + "=" * 60)
        print(f"Token Usage: {result['usage']['total_tokens']} tokens")


def demo_multi_step_research():
    """Demonstrate multi-step research workflow."""
    print("\n" + "=" * 60)
    print("Multi-Step Research Workflow")
    print("=" * 60)
    
    agent = DeepResearchAgent()
    
    # Step 1: Initial research
    print("\nStep 1: Conducting initial research...")
    print("-" * 60)
    
    research_result = agent.research(
        "What are the latest developments in quantum computing?",
        background=True
    )
    
    if research_result["status"] == "completed":
        print("\nInitial research completed!")
        
        # Step 2: Follow-up analysis using the research context
        print("\nStep 2: Analyzing findings...")
        print("-" * 60)
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        analysis = client.interactions.create(
            model="gemini-2.5-flash",
            input="Summarize the key findings in 3 bullet points.",
            previous_interaction_id=research_result["id"]
        )
        
        print("\nKey Findings:")
        print(analysis.outputs[-1].text)


def demo_comparative_research():
    """Demonstrate comparative research across multiple topics."""
    print("\n" + "=" * 60)
    print("Comparative Research")
    print("=" * 60)
    
    agent = DeepResearchAgent()
    
    topics = [
        "AI Safety and Alignment research in 2024",
        "Large Language Model developments in 2024"
    ]
    
    results = []
    
    for i, topic in enumerate(topics, 1):
        print(f"\nResearching Topic {i}: {topic}")
        print("-" * 60)
        
        result = agent.research(topic, background=True)
        results.append(result)
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        if result["status"] == "completed":
            print(f"\nTopic {i}:")
            print(f"Tokens used: {result['usage']['total_tokens']}")
            print(f"Report length: {len(result['report'])} characters")


class ResearchWorkflow:
    """Advanced research workflow with multiple stages."""
    
    def __init__(self):
        """Initialize the client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def comprehensive_research(self, topic: str) -> dict:
        """
        Conduct comprehensive research with multiple stages.
        
        Args:
            topic: Research topic
            
        Returns:
            Complete research package
        """
        print(f"Starting comprehensive research on: {topic}")
        print("=" * 60)
        
        # Stage 1: Deep Research
        print("\nStage 1: Deep Research")
        print("-" * 60)
        
        research_interaction = self.client.interactions.create(
            input=f"Research: {topic}",
            agent="deep-research-pro-preview-12-2025",
            background=True
        )
        
        # Poll for completion
        while True:
            interaction = self.client.interactions.get(research_interaction.id)
            print(f"Status: {interaction.status}")
            
            if interaction.status == "completed":
                break
            elif interaction.status in ["failed", "cancelled"]:
                return {"error": "Research failed"}
            
            time.sleep(10)
        
        research_report = interaction.outputs[-1].text
        
        # Stage 2: Extract Key Points
        print("\nStage 2: Extracting Key Points")
        print("-" * 60)
        
        key_points_interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input="Extract the 5 most important points from this research.",
            previous_interaction_id=research_interaction.id
        )
        
        key_points = key_points_interaction.outputs[-1].text
        
        # Stage 3: Generate Executive Summary
        print("\nStage 3: Generating Executive Summary")
        print("-" * 60)
        
        summary_interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input="Create a 2-paragraph executive summary.",
            previous_interaction_id=key_points_interaction.id
        )
        
        executive_summary = summary_interaction.outputs[-1].text
        
        return {
            "topic": topic,
            "full_report": research_report,
            "key_points": key_points,
            "executive_summary": executive_summary,
            "research_id": research_interaction.id
        }


def demo_comprehensive_workflow():
    """Demonstrate comprehensive research workflow."""
    print("\n" + "=" * 60)
    print("Comprehensive Research Workflow")
    print("=" * 60)
    
    workflow = ResearchWorkflow()
    
    result = workflow.comprehensive_research(
        "The impact of AI on healthcare in 2024-2025"
    )
    
    if "error" not in result:
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)
        print(result["executive_summary"])
        
        print("\n" + "=" * 60)
        print("KEY POINTS")
        print("=" * 60)
        print(result["key_points"])
        
        print("\n" + "=" * 60)
        print("FULL REPORT")
        print("=" * 60)
        print(result["full_report"][:500] + "...")  # Show first 500 chars


if __name__ == "__main__":
    main()
    
    # Uncomment to run additional demos
    # demo_multi_step_research()
    # demo_comparative_research()
    # demo_comprehensive_workflow()
    
    print("\n" + "=" * 60)
    print("Deep Research Agent Features")
    print("=" * 60)
    print("""
Key Capabilities:
- Autonomous research across multiple sources
- Long-running background execution
- Comprehensive report generation
- Stateful context for follow-up analysis

Best Practices:
1. Use background=True for long research tasks
2. Poll every 10-30 seconds to check status
3. Chain with regular models for post-processing
4. Store interaction IDs for later reference

Limitations:
- Only available with deep-research-pro-preview-12-2025 agent
- Requires background=True parameter
- May take several minutes to complete
- Beta feature - subject to changes
    """)
