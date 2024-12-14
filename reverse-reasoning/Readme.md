# Bidirectional Reasoning in LLMs

This repository implements bidirectional reasoning for Large Language Models, inspired by the research paper "Reverse Thinking Makes LLMs Stronger Reasoners" (UNC Chapel Hill & Google Research). The implementation demonstrates how LLMs can validate their reasoning by thinking both forward and backward, similar to human problem-solving processes.

## Extended Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI[User Interface]
        API[API Endpoints]
    end

    subgraph Core["Core Processing Layer"]
        direction TB
        QP[Question Processor]
        ES[Embedding Service]
        RS[Reasoning Service]
        VS[Verification Service]
    end

    subgraph Models["Model Layer"]
        GPT4[GPT-4 Teacher Model]
        ADA[ADA Embeddings Model]
        GPT4T[GPT-4 Turbo Student Model]
    end

    subgraph Storage["Storage Layer"]
        VStore[Vector Store]
        QStore[Question Store]
        RStore[Reasoning Chain Store]
    end

    subgraph External["External Services"]
        OpenAI[OpenAI API]
    end

    %% Client Layer Connections
    UI -->|HTTP Request| API
    API -->|Forward Request| QP

    %% Core Layer Internal Connections
    QP -->|Get Embeddings| ES
    QP -->|Process Question| RS
    RS -->|Verify Reasoning| VS
    ES -->|Store Vectors| VStore

    %% Model Layer Connections
    ES -->|Embedding Request| ADA
    RS -->|Forward Reasoning| GPT4
    RS -->|Generate Answer| GPT4T
    VS -->|Backward Reasoning| GPT4

    %% Storage Layer Connections
    VStore -->|Retrieve Similar| RS
    QStore -->|Store Questions| QP
    RStore -->|Store Chains| RS

    %% External Service Connections
    ADA -->|API Call| OpenAI
    GPT4 -->|API Call| OpenAI
    GPT4T -->|API Call| OpenAI

    %% Data Flow Annotations
    QP -->|"1. Process Input"| ES
    ES -->|"2. Get Embeddings"| VStore
    VStore -->|"3. Find Similar"| RS
    RS -->|"4. Forward Logic"| VS
    VS -->|"5. Verify & Store"| RStore

    classDef primary fill:#2563eb,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef secondary fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef storage fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff
    classDef external fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff

    class UI,API primary
    class QP,ES,RS,VS secondary
    class VStore,QStore,RStore storage
    class OpenAI external
    class GPT4,ADA,GPT4T external
```

## 🚀 Key Features

- Forward and backward reasoning paths
- Similarity-based case retrieval
- Consistency verification
- In-memory vector storage
- Configurable model selection (GPT-4, other compatible models)

## 💡 How It Works

The system implements a four-stage process for enhanced reasoning:

### 1. Input Processing
- Takes a question as input
- Generates embeddings using OpenAI's embedding model
- Finds similar cases from the knowledge base

### 2. Forward Reasoning
- Retrieves context from similar cases
- Generates step-by-step forward reasoning
- Produces an initial answer

### 3. Backward Verification
- Generates a reverse question from the answer
- Performs backward reasoning
- Verifies consistency between forward and backward paths

### 4. Result Processing
- Stores verified results in the knowledge base
- Updates similar cases database
- Discards inconsistent results

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bidirectional-reasoning.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Dependencies:
  - openai
  - python-dotenv
  - numpy
  - scipy
  - tenacity

## 🔧 Usage

```python
from reasoning_service import HealthcareReasoningService

# Initialize the service
service = HealthcareReasoningService()

# Process a question
async def ask_question():
    question = "Your medical question here"
    response = await service.answer_question(question)
    print(response)
```

## 🌟 Example

```python
# Example medical diagnosis question
question = "A 50-year-old patient presents with chest tightness and arm pain after exercise. What should be the immediate assessment?"

# Get response with both forward and backward reasoning
response = await service.answer_question(question)
```

## 🔄 Process Flow

1. **Question Input**
   - User submits a question
   - System generates embeddings
   - Similar cases are retrieved

2. **Forward Analysis**
   - Context compilation from similar cases
   - Step-by-step forward reasoning
   - Initial answer generation

3. **Backward Verification**
   - Generate reverse question
   - Perform backward reasoning
   - Verify consistency

4. **Result Handling**
   - Store verified results
   - Update knowledge base
   - Handle inconsistencies

## 🔍 Current Limitations

- In-memory storage (can be extended to vector databases)
- Limited to the context window of the underlying LLM
- Requires API calls for each reasoning step
- Performance depends on the quality of similar cases

## 🚀 Future Improvements

1. Integration with dedicated vector databases
2. Model fine-tuning for specific domains
3. Batch processing capabilities
4. Caching mechanism for frequent queries
5. Extended validation metrics
