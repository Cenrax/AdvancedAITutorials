"""
Function Calling Example
Demonstrates how to define custom tools and handle function calls.
"""

from google import genai
import os
from dotenv import load_dotenv
from typing import Dict, Any
import json

load_dotenv()


class FunctionCallingAgent:
    """Agent that can call custom functions."""
    
    def __init__(self):
        """Initialize the client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def execute_with_tools(self, prompt: str, tools: list, max_iterations: int = 5) -> str:
        """
        Execute a prompt with tool support, handling multiple tool calls.
        
        Args:
            prompt: User's request
            tools: List of tool definitions
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            Final response text
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=prompt,
            tools=tools
        )
        
        iterations = 0
        while iterations < max_iterations:
            # Check for function calls
            has_function_call = False
            
            for output in interaction.outputs:
                if output.type == "function_call":
                    has_function_call = True
                    print(f"\n🔧 Tool Call: {output.name}")
                    print(f"   Arguments: {json.dumps(output.arguments, indent=2)}")
                    
                    # Execute the function
                    result = self._execute_function(output.name, output.arguments)
                    print(f"   Result: {result}")
                    
                    # Send result back to model
                    interaction = self.client.interactions.create(
                        model="gemini-2.5-flash",
                        previous_interaction_id=interaction.id,
                        input=[{
                            "type": "function_result",
                            "name": output.name,
                            "call_id": output.id,
                            "result": str(result)
                        }]
                    )
            
            # If no function calls, we have the final answer
            if not has_function_call:
                break
            
            iterations += 1
        
        # Return final text response
        for output in interaction.outputs:
            if output.type == "text":
                return output.text
        
        return "No text response generated."
    
    def _execute_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a function by name.
        
        Args:
            name: Function name
            arguments: Function arguments
            
        Returns:
            Function result
        """
        # Map function names to actual implementations
        functions = {
            "get_weather": get_weather,
            "calculate_price": calculate_price,
            "search_database": search_database,
            "send_email": send_email,
            "get_stock_price": get_stock_price,
            "convert_currency": convert_currency
        }
        
        if name in functions:
            return functions[name](**arguments)
        else:
            return f"Error: Function '{name}' not found"


# Tool implementations
def get_weather(location: str) -> str:
    """Get weather for a location (mock implementation)."""
    # In production, this would call a real weather API
    weather_data = {
        "Paris": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "New York": "Rainy, 18°C",
        "Tokyo": "Clear, 25°C"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculate_price(base_price: float, tax_rate: float, discount: float = 0) -> Dict[str, float]:
    """Calculate final price with tax and discount."""
    discounted_price = base_price * (1 - discount / 100)
    tax_amount = discounted_price * (tax_rate / 100)
    final_price = discounted_price + tax_amount
    
    return {
        "base_price": base_price,
        "discount_amount": base_price - discounted_price,
        "tax_amount": tax_amount,
        "final_price": round(final_price, 2)
    }


def search_database(query: str, limit: int = 5) -> list:
    """Search a mock database."""
    # Mock database
    database = [
        {"id": 1, "name": "Product A", "price": 29.99},
        {"id": 2, "name": "Product B", "price": 49.99},
        {"id": 3, "name": "Product C", "price": 19.99},
        {"id": 4, "name": "Service A", "price": 99.99},
        {"id": 5, "name": "Service B", "price": 149.99}
    ]
    
    # Simple search by name
    results = [item for item in database if query.lower() in item["name"].lower()]
    return results[:limit]


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock implementation)."""
    # In production, this would actually send an email
    return f"Email sent to {to} with subject '{subject}'"


def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get stock price (mock implementation)."""
    # Mock stock data
    stocks = {
        "AAPL": {"price": 178.50, "change": 2.30, "change_percent": 1.31},
        "GOOGL": {"price": 142.80, "change": -1.20, "change_percent": -0.83},
        "MSFT": {"price": 378.90, "change": 5.60, "change_percent": 1.50}
    }
    return stocks.get(symbol, {"error": f"Stock {symbol} not found"})


def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Convert currency (mock implementation)."""
    # Mock exchange rates
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50
    }
    
    if from_currency not in rates or to_currency not in rates:
        return {"error": "Currency not supported"}
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates[from_currency]
    converted_amount = usd_amount * rates[to_currency]
    
    return {
        "original_amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted_amount": round(converted_amount, 2),
        "exchange_rate": round(rates[to_currency] / rates[from_currency], 4)
    }


# Tool definitions
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Gets the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., Paris, London, New York"
            }
        },
        "required": ["location"]
    }
}

PRICE_CALCULATOR_TOOL = {
    "type": "function",
    "name": "calculate_price",
    "description": "Calculates the final price including tax and optional discount.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_price": {
                "type": "number",
                "description": "The base price before tax and discount"
            },
            "tax_rate": {
                "type": "number",
                "description": "Tax rate as a percentage (e.g., 8.5 for 8.5%)"
            },
            "discount": {
                "type": "number",
                "description": "Discount percentage (optional, default 0)"
            }
        },
        "required": ["base_price", "tax_rate"]
    }
}

DATABASE_SEARCH_TOOL = {
    "type": "function",
    "name": "search_database",
    "description": "Searches the product database for items matching the query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)"
            }
        },
        "required": ["query"]
    }
}

EMAIL_TOOL = {
    "type": "function",
    "name": "send_email",
    "description": "Sends an email to a recipient.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line"
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            }
        },
        "required": ["to", "subject", "body"]
    }
}

STOCK_TOOL = {
    "type": "function",
    "name": "get_stock_price",
    "description": "Gets the current stock price and change for a given symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
            }
        },
        "required": ["symbol"]
    }
}

CURRENCY_TOOL = {
    "type": "function",
    "name": "convert_currency",
    "description": "Converts an amount from one currency to another.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "Amount to convert"
            },
            "from_currency": {
                "type": "string",
                "description": "Source currency code (e.g., USD, EUR, GBP)"
            },
            "to_currency": {
                "type": "string",
                "description": "Target currency code (e.g., USD, EUR, GBP)"
            }
        },
        "required": ["amount", "from_currency", "to_currency"]
    }
}


def main():
    """Demonstrate function calling capabilities."""
    print("=" * 60)
    print("Function Calling Examples")
    print("=" * 60)
    
    agent = FunctionCallingAgent()
    
    # Example 1: Single function call
    print("\n1. Single Function Call (Weather):")
    print("-" * 60)
    response = agent.execute_with_tools(
        "What's the weather in Paris?",
        tools=[WEATHER_TOOL]
    )
    print(f"\n✅ Final Response: {response}")
    
    # Example 2: Function with multiple parameters
    print("\n2. Function with Multiple Parameters (Price Calculator):")
    print("-" * 60)
    response = agent.execute_with_tools(
        "Calculate the final price for an item that costs $100 with 8.5% tax and a 10% discount.",
        tools=[PRICE_CALCULATOR_TOOL]
    )
    print(f"\n✅ Final Response: {response}")
    
    # Example 3: Multiple function calls in sequence
    print("\n3. Multiple Function Calls (Database Search + Price):")
    print("-" * 60)
    response = agent.execute_with_tools(
        "Search for products with 'Product' in the name, then calculate the price of the first result with 7% tax.",
        tools=[DATABASE_SEARCH_TOOL, PRICE_CALCULATOR_TOOL]
    )
    print(f"\n✅ Final Response: {response}")
    
    # Example 4: Stock price lookup
    print("\n4. Stock Price Lookup:")
    print("-" * 60)
    response = agent.execute_with_tools(
        "What's the current price of Apple stock (AAPL)?",
        tools=[STOCK_TOOL]
    )
    print(f"\n✅ Final Response: {response}")
    
    # Example 5: Currency conversion
    print("\n5. Currency Conversion:")
    print("-" * 60)
    response = agent.execute_with_tools(
        "Convert 100 USD to EUR.",
        tools=[CURRENCY_TOOL]
    )
    print(f"\n✅ Final Response: {response}")
    
    # Example 6: Multiple tools available
    print("\n6. Agent Chooses Appropriate Tool:")
    print("-" * 60)
    all_tools = [WEATHER_TOOL, PRICE_CALCULATOR_TOOL, STOCK_TOOL, CURRENCY_TOOL]
    response = agent.execute_with_tools(
        "What's the weather in Tokyo and what's the stock price of Microsoft?",
        tools=all_tools
    )
    print(f"\n✅ Final Response: {response}")


def demo_stateful_function_calling():
    """Demonstrate function calling with conversation context."""
    print("\n" + "=" * 60)
    print("Stateful Function Calling")
    print("=" * 60)
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # First interaction with tool
    print("\nTurn 1: Using tool")
    print("-" * 60)
    interaction1 = client.interactions.create(
        model="gemini-2.5-flash",
        input="What's the weather in London?",
        tools=[WEATHER_TOOL]
    )
    
    # Handle tool call
    for output in interaction1.outputs:
        if output.type == "function_call":
            result = get_weather(**output.arguments)
            interaction1 = client.interactions.create(
                model="gemini-2.5-flash",
                previous_interaction_id=interaction1.id,
                input=[{
                    "type": "function_result",
                    "name": output.name,
                    "call_id": output.id,
                    "result": result
                }]
            )
    
    print(f"Bot: {interaction1.outputs[-1].text}")
    
    # Follow-up without tool
    print("\nTurn 2: Follow-up question (no tool needed)")
    print("-" * 60)
    interaction2 = client.interactions.create(
        model="gemini-2.5-flash",
        input="Is that good weather for a picnic?",
        previous_interaction_id=interaction1.id,
        tools=[WEATHER_TOOL]
    )
    
    print(f"Bot: {interaction2.outputs[-1].text}")


if __name__ == "__main__":
    main()
    demo_stateful_function_calling()
