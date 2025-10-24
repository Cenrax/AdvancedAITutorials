from openai import OpenAI
from pydantic import BaseModel
from typing import List, Iterator
import json

client = OpenAI()


class ProductAnalysis(BaseModel):
    """Structured output for product analysis."""
    product_name: str
    features: List[str]
    pros: List[str]
    cons: List[str]
    price_range: str
    target_audience: str


class StreamingStructuredOutput:
    """Handler for streaming structured output."""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model
    
    def stream_analysis(self, product_description: str) -> Iterator[dict]:
        """Stream structured product analysis."""
        with self.client.responses.stream(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "Analyze the given product and provide structured output."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this product: {product_description}"
                }
            ],
            text_format=ProductAnalysis
        ) as stream:
            
            current_data = {
                "product_name": "",
                "features": [],
                "pros": [],
                "cons": [],
                "price_range": "",
                "target_audience": ""
            }
            
            for event in stream:
                if event.type == "response.output_text.delta":
                    try:
                        # Parse partial JSON updates
                        partial_data = json.loads(event.delta)
                        current_data.update(partial_data)
                        yield current_data.copy()
                    except json.JSONDecodeError:
                        continue
    
    def get_complete_analysis(self, product_description: str) -> ProductAnalysis:
        """Get complete structured analysis."""
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "Analyze the given product and provide structured output."
                },
                {
                    "role": "user",
                    "content": f"Analyze this product: {product_description}"
                }
            ],
            text_format=ProductAnalysis
        )
        
        return ProductAnalysis.model_validate_json(response.output_text)


def display_streaming_analysis(product: str):
    """Display streaming structured analysis."""
    handler = StreamingStructuredOutput()
    
    print(f"Analyzing: {product}")
    print("=" * 60)
    
    for update in handler.stream_analysis(product):
        print("\nCurrent Analysis:")
        for key, value in update.items():
            if value:  # Only show non-empty values
                print(f"  {key.replace('_', ' ').title()}: {value}")
        print("-" * 40)


def main():
    """Demonstrate structured output streaming."""
    product = "A smart home security camera with AI-powered motion detection, night vision, and cloud storage"
    
    print("Structured Output Streaming Demo")
    print("=" * 50)
    
    # Show streaming updates
    display_streaming_analysis(product)
    
    # Show final complete analysis
    print("\nComplete Analysis:")
    handler = StreamingStructuredOutput()
    complete = handler.get_complete_analysis(product)
    
    for field, value in complete.model_dump().items():
        print(f"{field.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()
