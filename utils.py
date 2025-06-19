# Import necessary modules
import backoff                 # For retrying operations with exponential backoff
import openai                  # OpenAI's SDK (used here for VLLM-compatible interface)
import asyncio                 # To run asynchronous tasks concurrently
from typing import Any         # For general typing
from retrying import retry     # Another retry library (used alongside backoff)
from pydantic import BaseModel
from typing import List
from pydantic import Field

# Initialize the OpenAI Async client to interact with the VLLM server
client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",   # Your locally hosted VLLM server endpoint
    api_key="token-abc123"               # API key used when running the VLLM server
)

# Retry chat completion request using backoff strategy for handling rate limit / network errors
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
async def completions_with_backoff(**kwargs):
    return await client.completions.create(**kwargs)

# Similar to above, but for chat-based completions
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
async def chat_completions_with_backoff(**kwargs):
    return await client.chat.completions.create(**kwargs) 


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
async def structured_output_completions_with_backoff(**kwargs):
    return await client.beta.chat.completions.parse(**kwargs)


class VLLM_MODEL:
    def __init__(self, model_name: str, stop_words: list[str], max_new_tokens: int) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words


    # Generate a chat response with retries (single input string)
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    async def chat_generate(self, input_string: str, temperature: float = 0.5, top_p: float = 1.0, solve: bool = False, response_format=None) -> tuple[str, str]:
        messages=[
                {"role": "system", "content": "You are a highly intelligent and logical assistant. Your goal is to solve deductional and logical reasoning tasks with clear, step-by-step explanations. Carefully analyze the given information, avoid making unstated assumptions, and provide a precise answer. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
                },
                {"role": "user", "content": input_string}
            ]
        if solve:
            response = await structured_output_completions_with_backoff(
                model=self.model_name,
                messages = messages,       
                temperature=temperature,
                top_p=top_p,
                stop=self.stop_words,
                response_format=response_format,
                max_tokens = 4096
                )
            
            output_object = response.choices[0].message.parsed
            # print(output_object)
            return output_object
        else:  
            response = await chat_completions_with_backoff(
                model=self.model_name,
                messages = messages,       
                temperature=temperature,
                top_p=top_p,
                stop=self.stop_words)
            # Extract the generated message and its stopping reason
            generated_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            return generated_text

    # Unified generate interface (currently uses chat-style by default)
    async def generate(self, input_string: str, temperature: float = 0.5, top_p: float = 1.0, solve: bool = False, response_format=None) -> tuple[str, str]:
        return await self.chat_generate(input_string, temperature, top_p, solve, response_format)

# ===============================
# Example usage (when running standalone script)
# Uncomment this to test manually
# ===============================
