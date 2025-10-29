"""
Anthropic Interleaved Thinking with Streaming
==============================================
Demonstrates real-time streaming of thinking blocks and tool calls.
This shows how to handle interleaved thinking in a streaming context.
"""

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def simulate_calculator(expression: str) -> str:
    """Simulate a calculator tool."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def simulate_database_query(query: str) -> str:
    """Simulate a database query."""
    mock_data = {
        "average monthly revenue": "5200",
        "total sales": "15",
        "product count": "42"
    }
    
    query_lower = query.lower()
    for key, value in mock_data.items():
        if key in query_lower:
            return value
    
    return "No matching data found"


# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

database_tool = {
    "name": "database_query",
    "description": "Query product database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query"
            }
        },
        "required": ["query"]
    }
}


def stream_with_interleaved_thinking():
    """Demonstrate streaming with interleaved thinking."""
    
    print("="*80)
    print("STREAMING INTERLEAVED THINKING DEMO")
    print("="*80)
    print("\nQuestion: Calculate 150 * 50 and compare to average monthly revenue\n")
    
    conversation_messages = [{
        "role": "user",
        "content": "What's the total revenue if we sold 150 units at $50 each, and how does this compare to our average monthly revenue?"
    }]
    
    # Track conversation state
    current_thinking = []
    current_tool_uses = []
    
    # First streaming request
    print("\n" + "─"*80)
    print("STREAMING STEP 1: Initial Analysis")
    print("─"*80 + "\n")
    
    with client.beta.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=[calculator_tool, database_tool],
        betas=["interleaved-thinking-2025-05-14"],
        messages=conversation_messages
    ) as stream:
        
        current_block_type = None
        current_text = ""
        
        for event in stream:
            # Content block start
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    current_block_type = "thinking"
                    print("💭 THINKING: ", end="", flush=True)
                elif event.content_block.type == "tool_use":
                    current_block_type = "tool_use"
                    tool_name = event.content_block.name
                    print(f"\n🔧 TOOL USE: {tool_name}")
                    print("   Input: ", end="", flush=True)
                elif event.content_block.type == "text":
                    current_block_type = "text"
                    print("💬 TEXT: ", end="", flush=True)
            
            # Content block delta (streaming content)
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                    current_text += event.delta.thinking
                elif event.delta.type == "input_json_delta":
                    print(event.delta.partial_json, end="", flush=True)
                    current_text += event.delta.partial_json
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    current_text += event.delta.text
            
            # Content block stop
            elif event.type == "content_block_stop":
                print()  # New line after block
                
                # Store the completed block
                if current_block_type == "thinking":
                    current_thinking.append({
                        "type": "thinking",
                        "thinking": current_text
                    })
                elif current_block_type == "tool_use":
                    # We'll get the full tool use from the final message
                    pass
                
                current_text = ""
                current_block_type = None
        
        # Get the final message
        final_message = stream.get_final_message()
        
        # Extract tool uses
        for block in final_message.content:
            if block.type == "tool_use":
                current_tool_uses.append(block)
    
    if not current_tool_uses:
        print("\n❌ No tool use detected")
        return
    
    # Execute ALL tools from the first response
    print(f"\n{'─'*80}")
    print(f"EXECUTING {len(current_tool_uses)} TOOL(S)")
    print(f"{'─'*80}\n")
    
    tool_results = []
    for tool in current_tool_uses:
        if tool.name == "calculator":
            result = simulate_calculator(tool.input["expression"])
            print(f"🔢 Calculator: {tool.input['expression']} = {result}")
        else:
            result = simulate_database_query(tool.input["query"])
            print(f"🗄️  Database: {tool.input['query']} = {result}")
        
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool.id,
            "content": result
        })
    
    # Add to conversation
    conversation_messages.append({
        "role": "assistant",
        "content": final_message.content
    })
    
    conversation_messages.append({
        "role": "user",
        "content": tool_results
    })
    
    # Second streaming request
    print(f"\n{'─'*80}")
    print("STREAMING STEP 2: After First Tool")
    print(f"{'─'*80}\n")
    
    second_tool_uses = []
    
    with client.beta.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=[calculator_tool, database_tool],
        betas=["interleaved-thinking-2025-05-14"],
        messages=conversation_messages
    ) as stream:
        
        current_block_type = None
        current_text = ""
        
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    current_block_type = "thinking"
                    print("💭 THINKING: ", end="", flush=True)
                elif event.content_block.type == "tool_use":
                    current_block_type = "tool_use"
                    print(f"\n🔧 TOOL USE: {event.content_block.name}")
                    print("   Input: ", end="", flush=True)
                elif event.content_block.type == "text":
                    current_block_type = "text"
                    print("💬 TEXT: ", end="", flush=True)
            
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "input_json_delta":
                    print(event.delta.partial_json, end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
            
            elif event.type == "content_block_stop":
                print()
        
        final_message_2 = stream.get_final_message()
        
        for block in final_message_2.content:
            if block.type == "tool_use":
                second_tool_uses.append(block)
    
    # If there are more tools, execute them
    if second_tool_uses:
        print(f"\n{'─'*80}")
        print(f"EXECUTING {len(second_tool_uses)} MORE TOOL(S)")
        print(f"{'─'*80}\n")
        
        tool_results_2 = []
        for tool in second_tool_uses:
            if tool.name == "calculator":
                result = simulate_calculator(tool.input["expression"])
                print(f"🔢 Calculator: {tool.input['expression']} = {result}")
            else:
                result = simulate_database_query(tool.input["query"])
                print(f"🗄️  Database: {tool.input['query']} = {result}")
            
            tool_results_2.append({
                "type": "tool_result",
                "tool_use_id": tool.id,
                "content": result
            })
        
        # Add to conversation
        conversation_messages.append({
            "role": "assistant",
            "content": final_message_2.content
        })
        
        conversation_messages.append({
            "role": "user",
            "content": tool_results_2
        })
        
        # Final streaming request
        print(f"\n{'─'*80}")
        print("STREAMING STEP 3: Final Analysis")
        print(f"{'─'*80}\n")
        
        with client.beta.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            tools=[calculator_tool, database_tool],
            betas=["interleaved-thinking-2025-05-14"],
            messages=conversation_messages
        ) as stream:
            
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        print("💭 THINKING: ", end="", flush=True)
                    elif event.content_block.type == "text":
                        print("💬 FINAL RESPONSE: ", end="", flush=True)
                
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        print(event.delta.thinking, end="", flush=True)
                    elif event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                
                elif event.type == "content_block_stop":
                    print()
    
    print(f"\n{'='*80}")
    print("STREAMING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        stream_with_interleaved_thinking()
    except anthropic.APIError as e:
        print(f"❌ Anthropic API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
