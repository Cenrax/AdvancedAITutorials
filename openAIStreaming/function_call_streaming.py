from openai import OpenAI
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def get_weather(location: str) -> Dict[str, Any]:
    """Mock weather function for demonstration."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%"
    }


tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Tokyo, Japan"
            }
        },
        "required": ["location"],
        "additionalProperties": False
    }
}]


def stream_function_call(query: str) -> Optional[Dict[str, Any]]:
    """Stream function call from OpenAI API."""
    stream = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": query}],
        tools=tools,
        stream=True
    )
    
    function_calls = {}
    
    for event in stream:
        if event.type == "response.output_item.added":
            if event.item.type == "function_call":
                function_calls[event.output_index] = {
                    "name": event.item.name,
                    "arguments": "",
                    "id": event.item.id
                }
                print(f"\n[EVENT: {event.type}] Function call started: {event.item.name}")
        
        elif event.type == "response.function_call_arguments.delta":
            index = event.output_index
            if index in function_calls:
                function_calls[index]["arguments"] += event.delta
                print(f"[{event.type}] {event.delta}", end="", flush=True)
        
        elif event.type == "response.function_call_arguments.done":
            index = event.output_index
            if index in function_calls:
                function_calls[index]["arguments"] = event.arguments
                print(f"\n[EVENT: {event.type}] Function call completed: {function_calls[index]['name']}")
                return json.loads(event.arguments)
    
    return None


def main():
    """Demonstrate function call streaming."""
    query = "What's the weather like in San Francisco today?"
    
    print("Streaming function call:")
    print("=" * 50)
    
    args = stream_function_call(query)
    
    if args:
        print("\n" + "=" * 50)
        print("Executing function with args:", args)
        result = get_weather(args["location"])
        print("Weather result:", result)


if __name__ == "__main__":
    main()
