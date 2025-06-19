# app.py
import re
import uvicorn
import logging
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from symbcot import initialize_symbcot_model
from vllm_utils import VLLM_MODEL

# Configure logging
logging.basicConfig(
    filename='api_requests.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request details
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        client_ip = request.client.host
        method = request.method
        url = str(request.url)
        
        # Get request body
        body = None
        if method in ["POST", "PUT"]:
            try:
                body = await request.body()
                body = json.loads(body)
            except:
                body = "Could not parse request body"
        
        # Log request
        log_dict = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "method": method,
            "url": url,
            "request_body": body
        }
        logging.info(f"REQUEST: {json.dumps(log_dict, indent=2)}")
        
        # Process the request and get response
        response = await call_next(request)
        
        # Get response body
        response_body = None
        if isinstance(response, JSONResponse):
            response_body = response.body.decode()
            try:
                response_body = json.loads(response_body)
            except:
                response_body = "Could not parse response body"
        
        # Log response
        response_log = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "status_code": response.status_code,
            "response_body": response_body
        }
        logging.info(f"RESPONSE: {json.dumps(response_log, indent=2)}")
        
        return response

# Add middleware to the app
app.add_middleware(RestrictDocsMiddleware)
app.add_middleware(RequestLoggingMiddleware)

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
    # Log the incoming query
    logging.info(f"QUERY INPUT: {json.dumps({
        'premises': query.premises_NL,
        'question': query.questions
    }, indent=2)}")

    result_dict = await process(query.premises_NL, query.questions)

    answer, idx, explanation = result_dict["Final_answer"], result_dict["idx"], result_dict["explanation"]
    idx = sorted(idx)

    final_dict_output = {
        "answers": answer,
        "idx": idx,
        "explanation": explanation
    }

    # Log the detailed answer
    logging.info(f"DETAILED ANSWER: {json.dumps({
        'question': query.questions,
        'answer': answer,
        'supporting_evidence_indices': idx,
        'explanation': explanation,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, indent=2)}")

    print("final_dict_output: ", final_dict_output)
    return final_dict_output

if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8080)
    except Exception as e:
        print(e)