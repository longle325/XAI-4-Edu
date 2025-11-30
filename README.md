# SymbCoT: Symbolic Chain-of-Thought Reasoning System

## XAI Challenge Competition Solution

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![vLLM](https://img.shields.io/badge/vLLM-Optimized-orange.svg)](https://github.com/vllm-project/vllm)


Developed as part of the Stubborn Strawberries Team for the XAI (Explainable AI) Challenge. The system handles logical reasoning, numerical computation, and multi-step inference while maintaining full explainability of its decision-making process.

## Overview

This project implements a multi-stage reasoning pipeline that processes complex logical questions through planning, execution, and verification stages. Built on top of FastAPI, it provides a production-ready REST API with comprehensive logging, error handling, and request validation middleware.

The inference engine uses vLLM for optimized model serving, achieving 2-3x speedup over standard implementations. It handles five distinct question types: Yes/No questions, multiple choice, numerical calculations, chained multi-part questions, and open-ended queries.

Every generated answer includes full explainability - the system tracks which premises were used and provides natural language justifications for its conclusions.

## Key Features

### Intelligent Question Classification

Questions are automatically categorized into five types:
- **YesNo**: Binary decision questions with uncertainty handling
- **MultiChoice**: Option-based questions (A/B/C/D)
- **Numerical**: Calculations, counts, and monetary values
- **ChainedQuestion**: Multi-part sequential reasoning
- **OpenEnded**: Descriptive and analytical responses

### Structured Reasoning Pipeline

The system follows a clear pipeline from input to answer:

```
Question Input → Classification → Plan Generation → Execution → Answer Extraction
```

Each stage is designed to be modular and traceable:
- **Symbolic Chain-of-Thought (SymbCoT)**: Breaks down complex reasoning into discrete, verifiable steps
- **Premise-Based Inference**: Uses only provided information with formal logical inference rules
- **Explainability**: Tracks which premises support each conclusion with full provenance

### Production-Ready API

The FastAPI service includes several production features:
- **Request Logging**: Comprehensive logging of all API interactions with timestamp and client information
- **Error Handling**: Retry mechanisms with exponential backoff for LLM API calls
- **Access Control**: Middleware for restricting documentation endpoints
- **Structured Responses**: Type-safe JSON responses using Pydantic models

## Technical Stack

### Core Technologies
- **Language**: Python 3.8+
- **LLM Framework**: vLLM (optimized inference server)
- **Models**: Qwen3-32B-AWQ, Qwen2.5-7B-Instruct
- **API Framework**: FastAPI + Uvicorn
- **Async Processing**: asyncio, aiohttp

### Key Libraries
```python
fastapi          # Web framework
vllm             # LLM inference optimization
pydantic         # Data validation and settings management
openai           # API client (vLLM-compatible endpoint)
backoff          # Retry mechanism with exponential backoff
```

## System Architecture

```
┌─────────────────┐
│  Client Request │
└────────┬────────┘
         │
┌────────▼────────────────────────┐
│   FastAPI Middleware Layer      │
│  - Request Logging              │
│  - Access Control               │
│  - Error Handling               │
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│   SymbCoT Reasoning Engine      │
│  ┌──────────────────────────┐  │
│  │ 1. Question Classifier   │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │ 2. Plan Generator        │  │
│  │    (Prompt Engineering)  │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │ 3. vLLM Executor         │  │
│  │    - Structured Output   │  │
│  │    - Retry Logic         │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │ 4. Answer Parser         │  │
│  │    - Extract Answer      │  │
│  │    - Extract Indices     │  │
│  │    - Extract Explanation │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
┌─────────────▼──────────────┐
│  Structured JSON Response  │
│  {                         │
│    "answers": "...",       │
│    "idx": [...],           │
│    "explanation": "..."    │
│  }                         │
└────────────────────────────┘
```

## Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU recommended for optimal performance
nvidia-smi
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/symbcot-xai
cd symbcot-xai

# Install dependencies
pip install -r requirements.txt

# Start vLLM server (in a separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --port 8000

# Start the API server
python backend_structured_output/main.py
```

### API Usage Example
```python
import requests

response = requests.post(
    "http://localhost:8080/query",
    json={
        "premises-NL": [
            "All students who complete required courses are eligible for graduation.",
            "John completed all required courses.",
            "John has a GPA above 3.5."
        ],
        "questions": "Is John eligible for graduation?"
    }
)

print(response.json())
# Output:
# {
#     "answers": "Yes",
#     "idx": [1, 2],
#     "explanation": "From Premise 1 and Premise 2, since John completed all 
#                     required courses, he is eligible for graduation using 
#                     If-Then Elimination (Modus Ponens)."
# }
```

## Implementation Details

### Structured Output Generation

The system uses Pydantic models to ensure type-safe responses from the LLM:

```python
class Response(BaseModel):
    explanation: str = Field(
        ..., 
        description='Natural language explanation logically justifying the answer'
    )
    idx: List[int] = Field(
        ..., 
        description='List of indices (1-based) of premises used'
    )
    Final_answer: str = Field(
        ..., 
        description='The final answer as a concise sentence or value'
    )
```

This approach ensures the model's output conforms to a specific schema, making parsing reliable and reducing errors.

### Multi-Step Logical Reasoning

The system implements 16 formal inference rules including:
- Modus Ponens / Modus Tollens
- Universal and Existential Instantiation
- Hypothetical Syllogism
- Disjunctive Syllogism
- And Introduction/Elimination
- Reductio ad Absurdum

Each reasoning step explicitly states which rule is being applied, creating a fully traceable inference chain.

### Verification System

An optional verification module can validate answers by:
- Cross-checking conclusions against all premises
- Exploring alternative answer possibilities
- Standardizing responses (e.g., normalizing "yes"/"true" to "Yes")
- Re-planning and solving if verification fails

## Performance Optimizations

Several techniques were employed to optimize system performance:

1. **Async Architecture**: All API endpoints use async/await for concurrent request handling
2. **vLLM Integration**: Achieves 2-3x faster inference compared to standard HuggingFace implementations
3. **AWQ Quantization**: Allows running 32B parameter model on 24GB VRAM with minimal quality loss
4. **Prompt Caching**: Common prompt patterns are reused to reduce processing overhead
5. **Exponential Backoff**: Intelligent retry strategy prevents API overload while ensuring reliability

## Testing and Evaluation

### Running Evaluations

```bash
# Run evaluation on test dataset
python evaluate.py \
    --dataset_name TRNS_AI \
    --model_name Qwen/Qwen3-32B-AWQ \
    --result_path ./results
```

### Metrics Tracked

The evaluation script computes:
- **Answer Accuracy**: Percentage of correct final answers
- **Premise Selection Metrics**: Precision, Recall, and F1 for selected premises
- **Exact Match**: Whether selected premises exactly match ground truth
- **Response Time**: Average time per question

## Project Structure

```
symbcot-xai/
├── backend_structured_output/
│   ├── main.py                 # FastAPI server with middleware
│   ├── symbcot.py              # Core reasoning engine
│   ├── vllm_utils.py           # LLM inference wrapper
│   └── verifier.py             # Answer verification module
├── prompts/
│   └── TRNS_AI/
│       ├── classify_question.txt
│       ├── ultimate_prompt.txt
│       └── [question_type]/
│           ├── plan_generation.txt
│           └── solver.txt
├── evaluate.py                 # Evaluation script
├── custom_symbcot.py           # Batch processing utilities
├── parser_utils.py             # Response parsing functions
└── requirements.txt
```

## Technical Decisions and Rationale

### Why vLLM?
vLLM provides significant performance improvements over standard inference methods through:
- Continuous batching
- PagedAttention for memory efficiency
- Optimized CUDA kernels
- Support for quantized models (AWQ, GPTQ)

### Why Structured Outputs?
Using Pydantic-validated structured outputs instead of regex parsing:
- Reduces parsing errors
- Makes output format explicit in the prompt
- Enables better model adherence to requirements
- Simplifies downstream processing

### Why Multi-Stage Pipeline?
Separating planning from execution:
- Improves reasoning quality through step-by-step breakdown
- Makes debugging easier (can inspect intermediate stages)
- Allows for different prompting strategies per stage
- Enables verification without re-running full pipeline

## Code Highlights

### Intelligent Retry Mechanism

```python
@retry(stop_max_attempt_number=3, wait_fixed=2000)
async def structured_output_chat_generate(
    self, 
    input_string: str, 
    temperature: float = 0.1, 
    output_structure = Response
):
    response = await structured_output_completions_with_backoff(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a highly intelligent..."},
            {"role": "user", "content": input_string}
        ],
        temperature=temperature,
        response_format=output_structure,
        max_tokens=self.max_new_tokens,
    )
    output_object = response.choices[0].message.reasoning_content
    return json.loads(output_object)
```

This pattern ensures reliability even when LLM API calls occasionally fail or timeout.

### Request Logging Middleware

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        client_ip = request.client.host
        
        # Log request details
        log_dict = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "method": request.method,
            "url": str(request.url),
            "request_body": await self._get_body(request)
        }
        logging.info(f"REQUEST: {json.dumps(log_dict, indent=2)}")
        
        response = await call_next(request)
        
        # Log response details
        logging.info(f"RESPONSE: {json.dumps(response_log, indent=2)}")
        
        return response
```

This provides complete audit trails for production debugging and monitoring.

## Skills Demonstrated

### AI/ML Engineering
- Large language model prompt engineering and optimization
- Structured output generation with schema validation
- Model serving optimization (vLLM, quantization techniques)
- Asynchronous programming patterns in Python

### Software Engineering
- RESTful API design with FastAPI
- Custom middleware implementation
- Robust error handling and retry mechanisms
- Production-grade logging and monitoring
- Clean code architecture with separation of concerns

### System Design
- Multi-stage processing pipelines
- Scalable microservice architecture
- Type-safe data validation with Pydantic
- Modular design for maintainability

### Domain Knowledge
- Formal logic and inference rule systems
- Symbolic reasoning approaches
- Explainable AI principles and practices
- Natural language processing

## Future Work

Several improvements could enhance the system:

- Implement multi-model ensemble voting for higher accuracy
- Fine-tune models on domain-specific logical reasoning datasets
- Add WebSocket support for streaming responses
- Containerize with Docker for easier deployment
- Create Kubernetes manifests for production scaling
- Integrate Prometheus metrics for monitoring
- Implement A/B testing framework for prompt optimization
- Add caching layer (Redis) for frequently asked questions

## Competition Context

This system was developed for the XAI (Explainable AI) Challenge, which focused on creating AI systems that can not only answer questions correctly but also explain their reasoning process in a human-understandable way. The competition emphasized:

- Logical correctness
- Premise tracking and citation
- Natural language explanations
- Handling diverse question types

Our team (Stubborn Strawberries) implemented a solution that balances performance with interpretability, making it suitable for real-world applications where trust and transparency are critical.


