# app.py
import re
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

#from answer_query import answerYesNo, answerMultiChoice, answerOpenEnded, answerChainedQuestion, answerNumeric
from symbcot import initialize_symbcot_model
from vllm_utils import VLLM_MODEL


class Query(BaseModel):
    premises_NL : List[str] = Field(..., alias="premises-NL")
    questions : str
    class Config:
        allow_population_by_field_name = True

app = FastAPI()

class RestrictDocsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the request is for /docs
        if request.url.path == "/docs":
            # Example: Allow only specific IP addresses
            allowed_ips = ["127.0.0.1", "::1"]  # Localhost IPs
            client_ip = request.client.host
            print(client_ip)
            if client_ip not in allowed_ips:
                raise HTTPException(status_code=403, detail="Access to /docs is restricted")
        return await call_next(request)

# Add middleware to the app
app.add_middleware(RestrictDocsMiddleware)

vllm_client = VLLM_MODEL(
    model_name="Qwen/Qwen3-32B-AWQ",
    max_new_tokens=10000,
    stop_words='-' * 100
)

symbcot_model = initialize_symbcot_model(vllm_client)

print("successfully init the vllm model!!!")


async def process(premises: list[str], question: str):
    return await symbcot_model.symbcot_reasoning_graph_generation(premises, question, None)

@app.get("/")
def read_root():
    return {"message": "Welcome to Stubborn Strawberries' API for XAI Challenge!"}

@app.post("/query")
async def query(query: Query):
    # answers, idxes, explanations = [], [], []

    result_dict = await process(query.premises_NL, query.questions)

    answer, idx, explanation = result_dict["Final_answer"], result_dict["idx"], result_dict["explanation"]
    idx = sorted(idx)

    final_dict_output = {
        "answers": answer,
        "idx" : idx,
        "explanation": explanation
    }
    
    print("final_dict_output: ", final_dict_output)

    return final_dict_output

if __name__ == "__main__":
    try:
        uvicorn.run("main_round2:app", host="0.0.0.0", port=8080)
    except Exception as e:
        print(e)