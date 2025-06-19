from vllm_utils import VLLM_MODEL    
from tqdm import tqdm    
import json
import asyncio
vllm_model = VLLM_MODEL(model_name= "Qwen/Qwen2.5-7B-Instruct",max_new_tokens=3500,stop_words='------')
def load_classification_prompt(question):
    file_path = '/root/trns-ai/SymbCoT/prompts/TRNS_AI/classify_question.txt'
    with open(file_path) as f:
        classification_prompt = f.read()
    full_classification_prompt = classification_prompt.replace("[[QUESTION]]", question)
    return full_classification_prompt
def classify_question(question):
    prompt_classify = load_classification_prompt(question)
    generated_text, finish_reason = asyncio.get_event_loop().run_until_complete(
        vllm_model.generate(prompt_classify, temperature=0.0)
    )
    return generated_text

if __name__ == "__main__":
    with open("/root/trns-ai/data/input.json","r") as f: 
        data = json.load(f)
    for item in data:
        questions = item.get("questions", [])
        for question in questions:
            typex = classify_question(question)
