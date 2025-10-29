# Anthropic Interleaved Thinking Demo

This demo showcases **Anthropic's Interleaved Thinking** feature, which allows Claude to think between tool calls for better decision-making and reasoning.

## What is Interleaved Thinking?

Interleaved thinking enables Claude to:
- **Think between tool calls** instead of only at the beginning
- **Analyze tool results** before deciding on next steps
- **Make more informed decisions** about which tools to use next
- **Provide better reasoning** throughout multi-step workflows

### Traditional vs Interleaved Thinking

**Traditional Tool Use:**
```
User Question → [Think] → Tool Call 1 → Tool Call 2 → Final Answer
```

**Interleaved Thinking:**
```
User Question → [Think] → Tool Call 1 → [Think] → Tool Call 2 → [Think] → Final Answer
```

## Features of This Demo

This example demonstrates:

1. **Multi-turn tool use** - Claude uses multiple tools sequentially
2. **Interleaved thinking blocks** - Thinking happens between each tool call
3. **Dynamic decision making** - Claude decides which tool to use based on previous results
4. **Clear visualization** - Each thinking block and tool call is clearly displayed

## Requirements

```bash
pip install anthropic python-dotenv
```

## Setup

1. Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your_api_key_here
```

2. Run the demo:
```bash
python main.py
```

## How It Works

### The Scenario

The demo asks Claude to:
1. Calculate revenue from selling 150 units at $50 each
2. Compare this to average monthly revenue from a database
3. Provide analysis

### The Flow

**Step 1: Initial Analysis**
- Claude receives the question
- Thinks about what information is needed
- Decides to use the calculator tool first

**Step 2: First Tool Execution**
- Calculator computes: 150 × 50 = 7500

**Step 3: Interleaved Thinking**
- Claude receives the calculator result
- **Thinks about what it means**
- Decides to query the database for comparison data

**Step 4: Second Tool Execution**
- Database returns average monthly revenue: $5200

**Step 5: Final Synthesis**
- Claude receives both results
- **Thinks about the comparison**
- Provides comprehensive analysis

## Code Structure

### Tool Definitions

```python
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "input_schema": {...}
}

database_tool = {
    "name": "database_query",
    "description": "Query product database",
    "input_schema": {...}
}
```

### Enabling Interleaved Thinking

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Max tokens for thinking
    },
    tools=[calculator_tool, database_tool],
    betas=["interleaved-thinking-2025-05-14"],  # Required beta flag
    messages=[...]
)
```

### Processing Response Blocks

The response contains different block types:

```python
for block in response.content:
    if block.type == "thinking":
        # Claude's internal reasoning
        print(block.thinking)
    elif block.type == "tool_use":
        # Tool call with parameters
        print(block.name, block.input)
    elif block.type == "text":
        # Final text response
        print(block.text)
```

## Key Configuration Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `model` | Claude model to use | `claude-sonnet-4-20250514` |
| `max_tokens` | Maximum response tokens | `16000` |
| `thinking.type` | Enable thinking mode | `"enabled"` |
| `thinking.budget_tokens` | Max tokens for thinking | `10000` |
| `betas` | Required beta feature flag | `["interleaved-thinking-2025-05-14"]` |

## Benefits of Interleaved Thinking

### 1. Better Tool Selection
Claude can analyze results before choosing the next tool, leading to more efficient workflows.

### 2. Improved Reasoning
Thinking between steps allows Claude to:
- Validate intermediate results
- Adjust strategy based on findings
- Provide more accurate final answers

### 3. Transparency
You can see Claude's reasoning process at each step, making it easier to:
- Debug issues
- Understand decisions
- Trust the output

### 4. Complex Workflows
Ideal for scenarios requiring:
- Multiple data sources
- Sequential decision-making
- Dynamic tool selection

## Example Output

```
================================================================================
  STEP 1: Initial Response - Claude Analyzes the Question
================================================================================

💭 THINKING:
The user wants me to calculate revenue and compare it to database data.
I should first calculate 150 × 50, then query the database.

🔧 TOOL USE: calculator
   Input: {'expression': '150 * 50'}

================================================================================
  STEP 2: Executing First Tool
================================================================================

🔢 Calculator result: 7500

================================================================================
  STEP 3: After First Tool - Interleaved Thinking
================================================================================

💭 THINKING:
The revenue is $7,500. Now I need to get the average monthly revenue
from the database to make a comparison.

🔧 TOOL USE: database_query
   Input: {'query': 'average monthly revenue'}

...
```

## Use Cases

This pattern is ideal for:

- **Data analysis workflows** - Fetch, calculate, compare
- **Research tasks** - Search, analyze, synthesize
- **Multi-step problem solving** - Break down complex questions
- **Decision support systems** - Gather info, evaluate, recommend

## Troubleshooting

### API Key Issues
Ensure your `.env` file contains a valid Anthropic API key:
```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Beta Feature Access
Interleaved thinking requires:
- Access to the beta feature
- The correct beta flag: `interleaved-thinking-2025-05-14`
- A compatible model: `claude-sonnet-4-20250514`

### Model Availability
If you get a model error, check that you have access to Claude Sonnet 4.

## Learn More

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)
- [Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)

## License

MIT
