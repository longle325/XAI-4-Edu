# Import necessary modules
import backoff                 # For retrying operations with exponential backoff
import openai                  # OpenAI's SDK (used here for VLLM-compatible interface)
from openai import RateLimitError, APIConnectionError  # Import specific error classes
import asyncio                 # To run asynchronous tasks concurrently
from typing import Any         # For general typing
from retrying import retry     # Another retry library (used alongside backoff)
from pydantic import BaseModel
from typing import List, Literal
from pydantic import Field
import json
# Initialize the OpenAI Async client to interact with the VLLM server
client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",   
    api_key="EMPTY"
)

class Response(BaseModel):
    explanation: str = Field(..., description='A natural language explanation logically justifying the final answer based on the premises.')
    idx: List[int] = Field(..., description='List of indices (1-based) of the premises used to justify the final answer.')
    Final_answer: str = Field(..., description='The final answer to the question, expressed as a concise sentence or paragraph in natural language. ')

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
async def structured_output_completions_with_backoff(**kwargs):
    return await client.beta.chat.completions.parse(**kwargs)

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
async def chat_completions_with_backoff(**kwargs):
    return await client.chat.completions.create(**kwargs)

class VLLM_MODEL:
    def __init__(self, model_name: str, stop_words: list[str], max_new_tokens: int) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    async def structured_output_chat_generate(self, input_string: str, temperature: float = 0.1, output_structure = Response) -> tuple[str, str]:
        response = await structured_output_completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a highly intelligent and logical assistant. Your goal is to solve deductional and logical reasoning tasks with clear, step-by-step explanations. Carefully analyze the given information, avoid making unstated assumptions, and provide a precise answer. Make sure you carefully and fully understand the details of given requirements and instructions before you start solving the problem."
                },
                {"role": "user", "content": input_string}
            ],
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words,
            response_format=output_structure,
            max_tokens=self.max_new_tokens,
        )
        output_object = response.choices[0].message.reasoning_content
        output_object = json.loads(output_object)
        
        return output_object

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    async def chat_generate(self, input_string: str, temperature: float = 0.0) -> tuple[str, str]:
        response = await chat_completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a highly intelligent and logical assistant. Your goal is to solve deductional and logical reasoning tasks with clear, step-by-step explanations. Carefully analyze the given information, avoid making unstated assumptions, and provide a precise answer. Make sure you carefully and fully understand the details of given requirements and instructions before you start solving the problem."
                },
                {"role": "user", "content": input_string}
            ],
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words,
            enable_thinking=False
        )
        # Extract the generated message and its stopping reason
        generated_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        return generated_text, finish_reason

    async def generate(self, input_string: str, temperature: float = 0.0) -> tuple[str, str]:
        return await self.chat_generate(input_string, temperature)

    async def structured_output_generate(self, input_string: str, temperature: float = 0.0, output_structure = Response) -> tuple[str, str]:
        return await self.structured_output_chat_generate(input_string, temperature, output_structure)

