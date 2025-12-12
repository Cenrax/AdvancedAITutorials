# Google Gemini Interactions API - Complete Guide

## What's New & Special About Interactions API?

The **Interactions API** represents a **paradigm shift** in how we work with AI models. Unlike traditional APIs that require you to manually manage every aspect of the conversation, the Interactions API is designed as a **high-level, stateful interface** that handles complexity for you.

### Key Innovations

#### 1. **Server-Side State Management** (Game Changer!)
- **Before**: You had to send the entire conversation history with every request
- **Now**: Just reference the previous interaction ID - the server remembers everything
- **Benefit**: Automatic context caching, reduced costs, simpler code

#### 2. **Unified Tool Orchestration**
- **Before**: Manual tool calling loops - you decide when to call tools, parse responses, handle errors
- **Now**: The model autonomously decides when to use tools and orchestrates multi-step workflows
- **Benefit**: Build agentic systems without complex orchestration logic

#### 3. **Background Execution for Long-Running Tasks**
- **Before**: Keep connections open or implement complex polling mechanisms
- **Now**: Start a task with `background=true` and poll for results when ready
- **Benefit**: Perfect for research agents, complex analysis, and time-intensive operations

#### 4. **Native Agent Support**
- **Before**: Only access to base models
- **Now**: Direct access to specialized agents like Deep Research that can autonomously perform complex tasks
- **Benefit**: Leverage pre-built agentic capabilities without building from scratch

#### 5. **Interaction as a First-Class Resource**
- **Before**: Conversations were just arrays of messages
- **Now**: Each interaction is a rich object with metadata, status, usage stats, and full history
- **Benefit**: Better observability, debugging, and conversation management

### What Makes It Special?

| Feature | Traditional API | Interactions API |
|---------|----------------|------------------|
| **State Management** | Client-side only | Server-side with automatic caching |
| **Tool Calling** | Manual loop implementation | Autonomous orchestration |
| **Long Tasks** | Complex async handling | Built-in background execution |
| **Conversation History** | Send full history each time | Reference by ID |
| **Agents** | Not available | Native support (Deep Research, etc.) |
| **Multimodal** | Separate endpoints | Unified interface |
| **Observability** | Limited | Rich metadata & status tracking |

### When to Use Interactions API?

**Perfect For:**
- Multi-turn conversational applications
- Agentic workflows with tool use
- Long-running research or analysis tasks
- Applications requiring conversation persistence
- Complex multimodal interactions

**Use Traditional API For:**
- Simple one-shot text generation
- Production apps requiring API stability (Interactions API is in Beta)
- When you need complete control over state management

---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Key Features](#key-features)
5. [Data Models](#data-models)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)
8. [Limitations](#limitations)

---

## Introduction

The **Interactions API** is a unified interface for interacting with Google's Gemini models and agents. It simplifies complex workflows by providing:

- **State Management**: Server-side conversation history tracking
- **Tool Orchestration**: Seamless integration with functions and built-in tools
- **Long-Running Tasks**: Background execution for complex operations
- **Multimodal Support**: Text, images, audio, video, and documents

### Why Use Interactions API?

Traditional API approaches require manual management of:
- Conversation history
- Tool execution loops
- State persistence
- Context caching

The Interactions API handles these automatically, letting you focus on building features.

---

## Core Concepts

### 1. Interaction Resource

An **Interaction** is the fundamental unit representing a complete turn in a conversation or task. It contains:

```
Interaction {
  id: string
  input: Content[]
  outputs: Content[]
  tools: Tool[]
  status: string
  usage: Usage
}
```

**Key Properties:**
- `id`: Unique identifier for state management
- `input`: User messages, tool results, or multimodal content
- `outputs`: Model responses (text, function calls, thoughts)
- `status`: `completed`, `in_progress`, `requires_action`, `failed`

### 2. State Management

**Stateful (Recommended):**
```python
# First interaction
interaction1 = client.interactions.create(
    model="gemini-2.5-flash",
    input="My name is Alice"
)

# Continue conversation
interaction2 = client.interactions.create(
    model="gemini-2.5-flash",
    input="What's my name?",
    previous_interaction_id=interaction1.id  # Links to previous context
)
```

**Benefits:**
- Automatic context caching
- Reduced token usage
- Simplified code

**Stateless:**
```python
# Manual history management
history = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "model", "content": "Hello Alice!"},
    {"role": "user", "content": "What's my name?"}
]

interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=history
)
```

### 3. Content Types

The API supports multiple content types in `input` and `outputs`:

| Type | Description | Example |
|------|-------------|---------|
| `text` | Plain text messages | `{"type": "text", "text": "Hello"}` |
| `image` | Images (base64 or URI) | `{"type": "image", "data": "...", "mime_type": "image/png"}` |
| `audio` | Audio files | `{"type": "audio", "data": "...", "mime_type": "audio/wav"}` |
| `video` | Video files | `{"type": "video", "uri": "...", "mime_type": "video/mp4"}` |
| `document` | PDFs and documents | `{"type": "document", "data": "...", "mime_type": "application/pdf"}` |
| `function_call` | Tool invocation | `{"type": "function_call", "name": "...", "arguments": {...}}` |
| `function_result` | Tool response | `{"type": "function_result", "result": "..."}` |

### 4. Tools

Three types of tools are available:

**a) Custom Functions:**
```python
weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Gets weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}
```

**b) Built-in Tools:**
- `google_search`: Grounding with Google Search
- `code_execution`: Execute Python code
- `url_context`: Fetch and analyze web pages

**c) MCP Servers:**
```python
mcp_server = {
    "type": "mcp_server",
    "name": "weather_service",
    "url": "https://example.com/mcp"
}
```

---

## Architecture

### Request Flow

```
User Input
    ↓
Interactions API
    ↓
Model Processing
    ↓
Tool Calls? ──Yes──→ Execute Tools ──→ Return Results ──→ Model Processing
    ↓ No
Final Response
```

### State Storage

**Default Behavior (`store=true`):**
- Interactions are stored server-side
- **Free Tier**: 1 day retention
- **Paid Tier**: 55 days retention

**Opt-out (`store=false`):**
- No server-side storage
- Cannot use `previous_interaction_id`
- Cannot use `background=true`

### Background Execution

For long-running tasks (e.g., Deep Research):

```python
# Start background task
interaction = client.interactions.create(
    agent="deep-research-pro-preview-12-2025",
    input="Research quantum computing advances",
    background=True
)

# Poll for completion
while True:
    status = client.interactions.get(interaction.id)
    if status.status == "completed":
        print(status.outputs[-1].text)
        break
    time.sleep(10)
```

---

## Key Features

### 1. Multimodal Understanding

**Image Analysis:**
```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "data": base64_image, "mime_type": "image/png"}
    ]
)
```

**Video Analysis:**
```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=[
        {"type": "text", "text": "Summarize this video"},
        {"type": "video", "data": base64_video, "mime_type": "video/mp4"}
    ]
)
```

### 2. Function Calling

**Automatic Tool Orchestration:**

The model decides when to call functions, and you handle the execution:

```python
# 1. Define tool
tools = [{
    "type": "function",
    "name": "calculate_price",
    "description": "Calculate total price with tax",
    "parameters": {...}
}]

# 2. Model calls tool
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="What's the total for $100 with 8% tax?",
    tools=tools
)

# 3. Execute and return result
for output in interaction.outputs:
    if output.type == "function_call":
        result = calculate_price(**output.arguments)
        
        interaction = client.interactions.create(
            model="gemini-2.5-flash",
            previous_interaction_id=interaction.id,
            input=[{
                "type": "function_result",
                "name": output.name,
                "call_id": output.id,
                "result": result
            }]
        )
```

### 3. Structured Output

Enforce JSON schema for reliable parsing:

```python
from pydantic import BaseModel

class ProductReview(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int
    summary: str

interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="Review: This product is amazing!",
    response_format=ProductReview.model_json_schema()
)

# Parse response
review = ProductReview.model_validate_json(interaction.outputs[-1].text)
```

### 4. Streaming

Receive responses incrementally:

```python
stream = client.interactions.create(
    model="gemini-2.5-flash",
    input="Explain machine learning",
    stream=True
)

for chunk in stream:
    if chunk.event_type == "content.delta":
        if chunk.delta.type == "text":
            print(chunk.delta.text, end="", flush=True)
```

### 5. Built-in Tools

**Google Search Grounding:**
```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="Who won the 2024 Super Bowl?",
    tools=[{"type": "google_search"}]
)
```

**Code Execution:**
```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="Calculate the 50th Fibonacci number",
    tools=[{"type": "code_execution"}]
)
```

**URL Context:**
```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="Summarize https://example.com",
    tools=[{"type": "url_context"}]
)
```

---

## Data Models

### Interaction Schema

```python
{
    "id": "interaction_abc123",
    "model": "gemini-2.5-flash",
    "input": [
        {"type": "text", "text": "Hello"}
    ],
    "outputs": [
        {"type": "text", "text": "Hi! How can I help?"}
    ],
    "tools": [],
    "previous_interaction_id": null,
    "status": "completed",
    "stream": false,
    "background": false,
    "store": true,
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    },
    "generation_config": {
        "temperature": 1.0,
        "max_output_tokens": 8192
    }
}
```

### Content Object

```python
{
    "type": "text" | "image" | "audio" | "video" | "document" | "function_call" | "function_result",
    "text": "...",           # For text
    "data": "...",          # Base64 encoded data
    "uri": "...",           # Remote URI
    "mime_type": "...",     # MIME type
    "name": "...",          # Function name
    "arguments": {...},     # Function arguments
    "result": "..."         # Function result
}
```

---

## Use Cases

### 1. Chatbots with Memory

```python
# User session
session_id = None

def chat(user_message):
    global session_id
    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input=user_message,
        previous_interaction_id=session_id
    )
    session_id = interaction.id
    return interaction.outputs[-1].text
```

### 2. Multi-Step Agents

```python
# Research → Summarize → Format
research = client.interactions.create(
    agent="deep-research-pro-preview-12-2025",
    input="Research AI trends",
    background=True
)

summary = client.interactions.create(
    model="gemini-2.5-flash",
    input="Summarize in 3 bullet points",
    previous_interaction_id=research.id
)

formatted = client.interactions.create(
    model="gemini-2.5-flash",
    input="Format as markdown",
    previous_interaction_id=summary.id
)
```

### 3. Data Extraction

```python
class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float
    items: List[dict]

interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=[
        {"type": "text", "text": "Extract invoice data"},
        {"type": "document", "data": pdf_base64, "mime_type": "application/pdf"}
    ],
    response_format=Invoice.model_json_schema()
)
```

### 4. Content Moderation

```python
class ModerationResult(BaseModel):
    is_safe: bool
    categories: List[str]
    confidence: float

interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=f"Moderate: {user_content}",
    response_format=ModerationResult.model_json_schema()
)
```

---

## Best Practices

### 1. Use Stateful Conversations

✅ **Do:**
```python
interaction2 = client.interactions.create(
    model="gemini-2.5-flash",
    input="Follow-up question",
    previous_interaction_id=interaction1.id
)
```

❌ **Don't:**
```python
# Manually reconstructing history each time
interaction2 = client.interactions.create(
    model="gemini-2.5-flash",
    input=[history1, history2, new_message]
)
```

**Why?** Stateful approach enables automatic caching, reducing latency and costs.

### 2. Set Appropriate Timeouts

```python
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input="Complex query",
    generation_config={
        "max_output_tokens": 2048,  # Limit response length
        "temperature": 0.7
    }
)
```

### 3. Handle Tool Calls Properly

```python
def handle_interaction(interaction):
    for output in interaction.outputs:
        if output.type == "function_call":
            # Execute tool
            result = execute_tool(output.name, output.arguments)
            
            # Return result
            return client.interactions.create(
                model="gemini-2.5-flash",
                previous_interaction_id=interaction.id,
                input=[{
                    "type": "function_result",
                    "name": output.name,
                    "call_id": output.id,
                    "result": result
                }]
            )
        elif output.type == "text":
            return output.text
```

### 4. Optimize File Handling

**For small files (<10MB):**
```python
# Use base64 inline
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=[{"type": "image", "data": base64_image, "mime_type": "image/png"}]
)
```

**For large files:**
```python
# Use Files API
file = client.files.upload(file="large_video.mp4")
interaction = client.interactions.create(
    model="gemini-2.5-flash",
    input=[{"type": "video", "uri": file.uri}]
)
```

### 5. Error Handling

```python
try:
    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input="Query"
    )
    
    if interaction.status == "failed":
        print(f"Error: {interaction.error}")
    elif interaction.status == "completed":
        print(interaction.outputs[-1].text)
        
except Exception as e:
    print(f"API Error: {e}")
```

### 6. Cost Optimization

- Use `gemini-2.5-flash-lite` for simple tasks
- Enable caching with `previous_interaction_id`
- Set `max_output_tokens` to limit response length
- Use `store=false` if you don't need state management

---

## Limitations

### Current Limitations (Beta)

1. **Unsupported Features:**
   - Grounding with Google Maps (coming soon)
   - Computer Use (coming soon)

2. **Tool Combinations:**
   - Cannot mix MCP + Function Calling + Built-in tools (coming soon)

3. **Model Support:**
   - Remote MCP not supported on Gemini 3 (coming soon)

4. **Output Ordering:**
   - Built-in tools may show text before tool execution (fix in progress)

### Breaking Changes Warning

⚠️ **The Interactions API is in Beta**

- Schemas may change
- SDK interfaces may evolve
- Feature behaviors may be updated

For production workloads, consider using the stable `generateContent` API until Interactions API reaches GA.

---

## Supported Models & Agents

| Name | Type | Model ID |
|------|------|----------|
| Gemini 2.5 Pro | Model | `gemini-2.5-pro` |
| Gemini 2.5 Flash | Model | `gemini-2.5-flash` |
| Gemini 2.5 Flash-lite | Model | `gemini-2.5-flash-lite` |
| Gemini 3 Pro Preview | Model | `gemini-3-pro-preview` |
| Deep Research | Agent | `deep-research-pro-preview-12-2025` |

---

## Configuration Options

### Generation Config

```python
generation_config = {
    "temperature": 0.7,           # Creativity (0.0-2.0)
    "max_output_tokens": 2048,    # Response length limit
    "top_p": 0.95,                # Nucleus sampling
    "top_k": 40,                  # Top-k sampling
    "thinking_level": "low"       # Reasoning verbosity
}
```

### Response Modalities

```python
# For image generation
interaction = client.interactions.create(
    model="gemini-3-pro-image-preview",
    input="Generate a sunset image",
    response_modalities=["IMAGE"]
)
```

---

## Summary

The Interactions API provides:

✅ **Simplified State Management** - No manual history tracking  
✅ **Tool Orchestration** - Automatic function calling  
✅ **Multimodal Support** - Text, images, audio, video, PDFs  
✅ **Background Execution** - Long-running tasks  
✅ **Structured Outputs** - Reliable JSON parsing  
✅ **Built-in Tools** - Search, code execution, URL context  

**Best For:**
- Conversational AI applications
- Multi-step agentic workflows
- Multimodal content analysis
- Data extraction and classification

**Use Stable API For:**
- Production workloads requiring stability
- Simple one-shot text generation
- Applications sensitive to breaking changes
