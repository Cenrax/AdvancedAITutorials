"""
Anthropic Interleaved Thinking Demo
====================================
Demonstrates how Claude can think between tool calls to make better decisions.
This example shows a multi-step workflow where Claude:
1. Calculates revenue from a sale
2. Queries a database for comparison
3. Provides analysis with interleaved thinking at each step
"""

import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def simulate_calculator(expression: str) -> str:
    """Simulate a calculator tool by evaluating simple expressions."""
    try:
        # Safe evaluation for simple math expressions
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def simulate_database_query(query: str) -> str:
    """Simulate a database query with mock data."""
    # Mock database responses
    mock_data = {
        "average monthly revenue": "5200",
        "total sales": "15",
        "product count": "42"
    }
    
    # Simple keyword matching for demo purposes
    query_lower = query.lower()
    for key, value in mock_data.items():
        if key in query_lower:
            return value
    
    return "No matching data found"


# Define tools
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '150 * 50')"
            }
        },
        "required": ["expression"]
    }
}

database_tool = {
    "name": "database_query",
    "description": "Query product database for sales and revenue information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to execute against the database"
            }
        },
        "required": ["query"]
    }
}


def print_separator(title: str = ""):
    """Print a visual separator for better readability."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def process_response_blocks(response, step_name: str):
    """Process and display response blocks with categorization."""
    print_separator(step_name)
    
    thinking_blocks = []
    tool_use_blocks = []
    text_blocks = []
    
    for block in response.content:
        if block.type == "thinking":
            thinking_blocks.append(block)
            print(f"💭 THINKING:\n{block.thinking}\n")
        elif block.type == "tool_use":
            tool_use_blocks.append(block)
            print(f"🔧 TOOL USE: {block.name}")
            print(f"   Input: {block.input}\n")
        elif block.type == "text":
            text_blocks.append(block)
            print(f"💬 TEXT:\n{block.text}\n")
    
    return thinking_blocks, tool_use_blocks, text_blocks


def main():
    """Main demonstration of interleaved thinking with multi-turn tool use."""
    
    print_separator("INTERLEAVED THINKING DEMONSTRATION")
    print("Question: What's the total revenue if we sold 150 units of product A at $50 each,")
    print("and how does this compare to our average monthly revenue from the database?")
    
    # Store conversation history
    conversation_messages = []
    all_thinking_blocks = []
    all_tool_use_blocks = []
    
    # Initial user message
    user_message = {
        "role": "user",
        "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }
    conversation_messages.append(user_message)
    
    # ========================================================================
    # STEP 1: Initial request - Claude thinks and decides which tool to use
    # ========================================================================
    
    response1 = client.beta.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=[calculator_tool, database_tool],
        betas=["interleaved-thinking-2025-05-14"],
        messages=conversation_messages
    )
    
    thinking_blocks, tool_use_blocks, text_blocks = process_response_blocks(
        response1, 
        "STEP 1: Initial Response - Claude Analyzes the Question"
    )
    
    all_thinking_blocks.extend(thinking_blocks)
    all_tool_use_blocks.extend(tool_use_blocks)
    
    if not tool_use_blocks:
        print("❌ No tool use detected. Exiting.")
        return
    
    # Add assistant's response to conversation (include ALL blocks)
    assistant_content_1 = []
    if thinking_blocks:
        assistant_content_1.extend(thinking_blocks)
    if tool_use_blocks:
        assistant_content_1.extend(tool_use_blocks)
    
    conversation_messages.append({
        "role": "assistant",
        "content": assistant_content_1
    })
    
    # ========================================================================
    # STEP 2: Execute ALL tools from first response and provide results
    # ========================================================================
    
    print_separator(f"STEP 2: Executing {len(tool_use_blocks)} Tool(s)")
    
    tool_results = []
    for tool in tool_use_blocks:
        if tool.name == "calculator":
            result = simulate_calculator(tool.input["expression"])
            print(f"🔢 Calculator: {tool.input['expression']} = {result}")
        elif tool.name == "database_query":
            result = simulate_database_query(tool.input["query"])
            print(f"🗄️  Database: {tool.input['query']} = {result}")
        else:
            result = "Unknown tool"
            print(f"❓ Unknown tool: {tool.name}")
        
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool.id,
            "content": result
        })
    
    # Add all tool results to conversation
    conversation_messages.append({
        "role": "user",
        "content": tool_results
    })
    
    # ========================================================================
    # STEP 3: Claude receives first result, thinks, and may use another tool
    # ========================================================================
    
    response2 = client.beta.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=[calculator_tool, database_tool],
        betas=["interleaved-thinking-2025-05-14"],
        messages=conversation_messages
    )
    
    thinking_blocks_2, tool_use_blocks_2, text_blocks_2 = process_response_blocks(
        response2,
        "STEP 3: After First Tool - Interleaved Thinking"
    )
    
    all_thinking_blocks.extend(thinking_blocks_2)
    all_tool_use_blocks.extend(tool_use_blocks_2)
    
    # Add assistant's response to conversation
    assistant_content_2 = []
    if thinking_blocks_2:
        assistant_content_2.extend(thinking_blocks_2)
    if tool_use_blocks_2:
        assistant_content_2.extend(tool_use_blocks_2)
    if text_blocks_2 and not tool_use_blocks_2:
        # If there's text and no more tools, we're done
        assistant_content_2.extend(text_blocks_2)
    
    if tool_use_blocks_2:
        conversation_messages.append({
            "role": "assistant",
            "content": assistant_content_2
        })
        
        # ====================================================================
        # STEP 4: Execute additional tools if needed
        # ====================================================================
        
        print_separator(f"STEP 4: Executing {len(tool_use_blocks_2)} Additional Tool(s)")
        
        tool_results_2 = []
        for tool in tool_use_blocks_2:
            if tool.name == "calculator":
                result = simulate_calculator(tool.input["expression"])
                print(f"🔢 Calculator: {tool.input['expression']} = {result}")
            elif tool.name == "database_query":
                result = simulate_database_query(tool.input["query"])
                print(f"🗄️  Database: {tool.input['query']} = {result}")
            else:
                result = "Unknown tool"
                print(f"❓ Unknown tool: {tool.name}")
            
            tool_results_2.append({
                "type": "tool_result",
                "tool_use_id": tool.id,
                "content": result
            })
        
        # Add all tool results to conversation
        conversation_messages.append({
            "role": "user",
            "content": tool_results_2
        })
        
        # ====================================================================
        # STEP 5: Final response with all information
        # ====================================================================
        
        response3 = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            tools=[calculator_tool, database_tool],
            betas=["interleaved-thinking-2025-05-14"],
            messages=conversation_messages
        )
        
        thinking_blocks_3, tool_use_blocks_3, text_blocks_3 = process_response_blocks(
            response3,
            "STEP 5: Final Response - Synthesis and Analysis"
        )
    else:
        # No second tool needed, response2 contains final answer
        print_separator("FINAL RESPONSE (No Second Tool Needed)")
        if text_blocks_2:
            for text_block in text_blocks_2:
                print(f"💬 {text_block.text}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print_separator("SUMMARY")
    print(f"✅ Total thinking blocks: {len(all_thinking_blocks)}")
    print(f"✅ Total tool calls: {len(all_tool_use_blocks)}")
    print(f"✅ Tools used: {[tool.name for tool in all_tool_use_blocks]}")
    print("\n🎯 Key Benefit of Interleaved Thinking:")
    print("   Claude can think between tool calls, allowing it to:")
    print("   - Analyze results before deciding on next steps")
    print("   - Make more informed decisions about which tools to use")
    print("   - Provide better reasoning and explanations")
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except anthropic.APIError as e:
        print(f"❌ Anthropic API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
