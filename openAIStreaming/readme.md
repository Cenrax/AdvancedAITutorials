# OpenAI Streaming API Guide

This repository provides comprehensive examples of streaming with the OpenAI Responses API. Streaming allows you to receive partial results as they're generated, improving user experience and reducing perceived latency.

## What is Streaming?

Streaming is a technique where the OpenAI API sends back response chunks as they're generated, rather than waiting for the complete response. This provides several benefits:

- **Real-time feedback**: Users see content immediately as it's generated
- **Reduced latency**: First tokens appear faster
- **Better UX**: Applications feel more responsive
- **Progress indication**: Users can see the model is working
- **Memory efficiency**: Process chunks without storing entire response

## Repository Structure

```
openaiplaybook/
├── basic_streaming.py          # Simple text streaming
├── function_call_streaming.py  # Function call streaming
├── structured_streaming.py     # Structured output streaming
├── requirements.txt            # Dependencies
└── README.md                   # This guide
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 1. Basic Text Streaming

File: `basic_streaming.py`

Demonstrates simple text generation with streaming enabled.

### Key Concepts
- **Streaming Events**: The API emits different event types during generation
- **Text Deltas**: Small chunks of text as they're generated
- **Event Filtering**: Only process events you care about

### Usage
```python
from basic_streaming import stream_text_response

# Stream a simple response
for chunk in stream_text_response("Write a haiku about AI"):
    print(chunk, end="", flush=True)
```

### Event Types
- `response.created`: Response initialization
- `response.output_text.delta`: New text chunk available
- `response.completed`: Generation finished
- `error`: Error occurred

## 2. Function Call Streaming

File: `function_call_streaming.py`

Shows how to stream function calls and their arguments as they're built.

### Key Concepts
- **Function Detection**: Identify when model wants to call a function
- **Argument Streaming**: Watch arguments being built in real-time
- **Tool Integration**: Connect streamed calls to actual functions

### Usage
```python
from function_call_streaming import stream_function_call

# Stream a function call
result = stream_function_call("What's the weather in Tokyo?")
print(f"Function arguments: {result}")
```

### Event Flow
1. `response.output_item.added`: Function call detected
2. `response.function_call_arguments.delta`: Arguments chunk received
3. `response.function_call_arguments.done`: Arguments complete
4. `response.output_item.done`: Function call finalized

### Argument Accumulation
```python
final_tool_calls = {}

for event in stream:
    if event.type == 'response.output_item.added':
        final_tool_calls[event.output_index] = event.item
    elif event.type == 'response.function_call_arguments.delta':
        index = event.output_index
        final_tool_calls[index].arguments += event.delta
```

## 3. Structured Output Streaming

File: `structured_streaming.py`

Demonstrates streaming structured data using Pydantic models.

### Key Concepts
- **Schema Validation**: Ensure streaming data matches expected structure
- **Partial Updates**: Handle incomplete JSON during streaming
- **Type Safety**: Use Pydantic for automatic validation

### Usage
```python
from structured_streaming import StreamingStructuredOutput

handler = StreamingStructuredOutput()

# Stream structured analysis
for update in handler.stream_analysis("Smart home security camera"):
    print(f"Current features: {update.get('features', [])}")
```

### Pydantic Model Example
```python
from pydantic import BaseModel
from typing import List

class ProductAnalysis(BaseModel):
    product_name: str
    features: List[str]
    pros: List[str]
    cons: List[str]
    price_range: str
    target_audience: str
```

## Advanced Streaming Patterns

### 1. Real-time Progress Updates
```python
def stream_with_progress(prompt):
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    buffer = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            buffer += event.delta
            # Update progress every 50 characters
            if len(buffer) % 50 == 0:
                yield {"type": "progress", "content": buffer}
        elif event.type == "response.completed":
            print("Completed")
            # print(event.response.output)

    final_response = stream.get_final_response()
    print(final_response)
```

### 2. Multi-turn Streaming
```python
def stream_conversation(messages):
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        stream=True
    )
    
    response_text = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            response_text += event.delta
            yield {
                "partial": response_text,
                "delta": event.delta
            }
```

### 3. Error Handling
```python
def safe_stream(prompt):
    try:
        stream = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for event in stream:
            if event.type == "error":
                yield {"type": "error", "message": event.error.message}
                break
            elif event.type == "response.output_text.delta":
                yield {"type": "content", "data": event.delta}
                
    except Exception as e:
        yield {"type": "error", "message": str(e)}
```

## Performance Considerations

### Latency Benefits
- **First token latency**: Streaming reduces time to first token by 20-40%
- **Perceived performance**: Users see progress immediately
- **Network efficiency**: Smaller packets reduce buffering

### Memory Usage
- **Streaming**: Constant memory usage (only current chunk)
- **Non-streaming**: Memory scales with response length

### Rate Limits
- Streaming requests count the same as non-streaming
- Each chunk doesn't count as a separate request
- Token usage calculated on complete response

## Best Practices

### 1. Event Filtering
```python
# Only process text deltas
for event in stream:
    if event.type == "response.output_text.delta":
        process_text(event.delta)
    elif event.type == "error":
        handle_error(event.error)
```

### 2. Buffer Management
```python
# Accumulate text with buffer
buffer = ""
for event in stream:
    if event.type == "response.output_text.delta":
        buffer += event.delta
        if len(buffer) > 100:  # Process in chunks
            yield buffer
            buffer = ""
```

### 3. Connection Management
```python
# Use context manager for proper cleanup
with client.responses.stream(...) as stream:
    for event in stream:
        process_event(event)
```

### 4. Error Recovery
```python
def resilient_stream(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            stream = client.responses.create(...)
            for event in stream:
                yield event
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Testing Your Streaming

### 1. Basic Test
```bash
python basic_streaming.py
```

### 2. Function Call Test
```bash
python function_call_streaming.py
```

### 3. Structured Output Test
```bash
python structured_streaming.py
```

## Troubleshooting

### Common Issues

**No events received**
- Check API key configuration
- Verify model availability
- Check network connectivity

**Partial JSON errors in structured streaming**
- Handle JSON decode errors gracefully
- Use try-except blocks for parsing
- Accumulate until valid JSON is received

**Slow streaming**
- Check network latency
- Try different models (gpt-4o-mini is faster)
- Reduce prompt complexity

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for OpenAI
import openai
openai.log = "debug"
```

## API Reference

### Streaming Parameters
- `stream=True`: Enable streaming mode
- `model`: Any OpenAI model (gpt-4o, gpt-4o-mini, etc.)
- `tools`: List of available functions
- `text_format`: Pydantic model for structured output

### Event Types
- `response.created`: Response initialized
- `response.output_text.delta`: Text chunk
- `response.completed`: Response finished
- `response.function_call_arguments.delta`: Function argument chunk
- `response.function_call_arguments.done`: Function arguments complete
- `error`: Error occurred

## Next Steps

1. **Experiment**: Try different models and prompts
2. **Integrate**: Add streaming to your applications
3. **Optimize**: Implement caching and connection pooling
4. **Monitor**: Add logging and metrics for production use

## Resources

- [OpenAI Streaming Documentation](https://platform.openai.com/docs/guides/streaming)
- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python OpenAI SDK](https://github.com/openai/openai-python)
