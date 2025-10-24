from openai import OpenAI
from typing import Iterator
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def stream_text_response(prompt: str, model: str = "gpt-4.1-mini") -> Iterator[str]:
    """Stream text response from OpenAI API."""
    stream = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
        elif event.type == "response.completed":
            break


def main():
    """Demonstrate basic text streaming."""
    prompt = "Write a haiku about artificial intelligence."
    
    print("Streaming response:")
    print("-" * 50)
    
    for chunk in stream_text_response(prompt):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 50)
    print("Streaming complete!")


if __name__ == "__main__":
    main()
