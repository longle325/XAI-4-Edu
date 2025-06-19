#!/usr/bin/env python3
import json
import time
import asyncio
import os
import re
from datetime import datetime

# Import necessary modules
import backoff
import openai
from typing import Any
from retrying import retry

# Initialize OpenAI clients for different models
qwen25_client = openai.AsyncOpenAI(
    base_url="http://localhost:7999/v1",
    api_key="token-abc123"
)

qwen3_client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

# Retry chat completion request using backoff strategy
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
async def chat_completions_with_backoff(client, **kwargs):
    return await client.chat.completions.create(**kwargs)

class CustomVLLMModel:
    def __init__(self, model_name: str, max_new_tokens: int, stop_words: str, client) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        self.client = client

    # Generate a chat response with retries
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    async def generate(self, input_string: str, temperature: float = 0.0) -> tuple[str, str]:
        messages=[
            {"role": "system", "content": "You are a highly intelligent and logical assistant."},
            {"role": "user", "content": input_string}
        ]
        
        try:
            response = await chat_completions_with_backoff(
                self.client,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                stop=self.stop_words,
                max_tokens=self.max_new_tokens
            )
            
            # Check if response has the expected structure
            if (hasattr(response, 'choices') and 
                len(response.choices) > 0 and 
                hasattr(response.choices[0], 'message') and 
                hasattr(response.choices[0].message, 'content') and
                response.choices[0].message.content is not None):
                
                generated_text = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason
                return generated_text, finish_reason
            else:
                # Handle unexpected response structure
                print(f"Warning: Unexpected response structure: {response}")
                return f"Error: Could not generate response for the given input.", "api_error"
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}", "error"

# Initialize models
qwen25_model = CustomVLLMModel(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=3500,
    stop_words='------',
    client=qwen25_client
)

qwen3_model = CustomVLLMModel(
    model_name="Qwen/Qwen3-32B-AWQ",
    max_new_tokens=20000,
    stop_words='------',
    client=qwen3_client
)

def load_classification_prompt(question):
    """Load the classification prompt and insert the question."""
    file_path = '/root/trns-ai/st_CoT/prompts/TRNS_AI/classify_question.txt'
    with open(file_path) as f:
        classification_prompt = f.read()
    return classification_prompt.replace("[[QUESTION]]", question)

def load_plan_generation_prompt(context, question):
    # premises = ""
    # for i, premise in enumerate(context):
    #     premises = premises + f"{i+1}." + premise + "\n"
    sentences = [s.strip() for s in context.split(". ") if context.strip()]
    premises = "\n".join([f"{i+1}.{sentence}." for i, sentence in enumerate(sentences)])
    """Load the plan generation prompt and insert context and question."""
    file_path = '/root/trns-ai/st_CoT/prompts/TRNS_AI/ultimate_prompt.txt'
    with open(file_path) as f:
        plan_prompt = f.read()
    print(f"Premises: {premises}")
    plan_prompt = plan_prompt.replace("[[CONTEXT]]", premises)
    plan_prompt = plan_prompt.replace("[[QUESTION]]", question)
    return plan_prompt

def extract_answer(raw_response):
    """Extract the final answer from the model's response."""
    # Try to find the answer pattern: "Final answer: X"
    answer_match = re.search(r'Final answer:\s*(.*?)(?:\n|$)', raw_response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # If no specific pattern is found, look for lines after numbered steps
    lines = raw_response.strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith("Final answer:"):
            return line.replace("Final answer:", "").strip()
    
    # Last resort: look for the answer after numbered reasoning steps
    for i, line in enumerate(lines):
        if re.match(r'^\d+\.', line) and i+1 < len(lines):
            # Found a numbered step, check the following lines for an answer
            for j in range(i+1, len(lines)):
                if not re.match(r'^\d+\.', lines[j]) and lines[j].strip() and not lines[j].startswith("idx:") and not lines[j].startswith("explanation:"):
                    return lines[j].strip()
    
    # If nothing found, return the raw response (truncated if too long)
    return raw_response[:100] + "..." if len(raw_response) > 100 else raw_response

def extract_indices(raw_response):
    """Extract the indices used from the model's response."""
    idx_match = re.search(r'idx:\s*\[(.*?)\]', raw_response, re.IGNORECASE)
    if idx_match:
        try:
            # Extract comma-separated numbers and convert to integers
            idx_str = idx_match.group(1).strip()
            if idx_str:
                return [int(idx.strip()) for idx in idx_str.split(',')]
        except ValueError:
            pass
    
    # Try alternative formats if the expected format is not found
    idx_line = re.search(r'idx:\s*(.*?)(?:\n|$)', raw_response, re.IGNORECASE)
    if idx_line:
        try:
            # Try to parse a comma-separated list without brackets
            idx_str = idx_line.group(1).strip()
            if idx_str:
                # Remove any non-numeric characters except commas and spaces
                cleaned_str = re.sub(r'[^\d,\s]', '', idx_str)
                return [int(idx.strip()) for idx in cleaned_str.split(',') if idx.strip()]
        except ValueError:
            pass
    
    return []

def extract_explanation(raw_response):
    """Extract the explanation from the model's response."""
    explanation_match = re.search(r'explanation:\s*(.*?)(?:\n\n|$)', raw_response, re.IGNORECASE | re.DOTALL)
    if explanation_match:
        return explanation_match.group(1).strip()
    return ""

def compare_indices(predicted_indices, ground_truth_indices):
    """Compare predicted indices with ground truth indices."""
    if not predicted_indices or not ground_truth_indices:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False
        }
    
    # Convert to sets for comparison
    pred_set = set(predicted_indices)
    gt_set = set(ground_truth_indices)
    
    # Calculate metrics
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gt_set) if gt_set else 0
    
    # Calculate F1 score
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": pred_set == gt_set
    }

def compare_answers(predicted, ground_truth):
    """Compare the predicted answer with the ground truth."""
    # Normalize both answers for comparison
    pred_normalized = predicted.strip().lower()
    gt_normalized = ground_truth.strip().lower()
    print(f"Predict: {pred_normalized}")
    print(f"Ground Truth: {gt_normalized}")
    # Check for exact match
    if pred_normalized == gt_normalized:
        return True

    # Check for numeric match (if both are numbers)
    try:
        # Handle currency format
        pred_numeric = pred_normalized.replace('$', '').replace(',', '')
        gt_numeric = gt_normalized.replace('$', '').replace(',', '')
        
        # Convert to float and compare with a small tolerance
        pred_float = float(pred_numeric)
        gt_float = float(gt_numeric)
        
        # Allow small floating point differences
        return abs(pred_float - gt_float) < 0.01
    except ValueError:
        # If conversion fails, they're not both numbers
        pass
    
    return False

async def process_question(item):
    """Process a single question through the pipeline."""
    question = item["question"]
    print(f"Processing question: {question}")
    context = item["context"]
    question_id = item["id"]
    ground_truth = item["answer"]
    ground_truth_indices = item.get("idx", [])
    
    result = {
        "id": question_id,
        "question": question,
        "context": context,
        "ground_truth": ground_truth,
        "ground_truth_indices": ground_truth_indices
    }
    
    # Generate answer using Qwen 3 and plan_generation.txt
    try:
        start_time = time.time()
        plan_prompt = load_plan_generation_prompt(context, question)
        
        # Print prompt length for debugging
        print(f"Prompt length: {len(plan_prompt)} characters")
        
        raw_answer, finish_reason = await qwen3_model.generate(plan_prompt, temperature=0.0)
        answer_time = time.time() - start_time
        print(f"Answer time: {answer_time} seconds, Finish reason: {finish_reason}")
        
        if finish_reason == "error" or finish_reason == "api_error":
            print(f"Warning: Received error in response. Answer: {raw_answer}")
        
        # Extract the final answer from the raw response
        extracted_answer = extract_answer(raw_answer)
        extracted_indices = extract_indices(raw_answer)
        extracted_explanation = extract_explanation(raw_answer)
        
        # Compare with ground truth
        is_correct = compare_answers(extracted_answer, ground_truth)
        indices_comparison = compare_indices(extracted_indices, ground_truth_indices)
        
        result["raw_answer"] = raw_answer
        result["extracted_answer"] = extracted_answer
        result["extracted_indices"] = extracted_indices
        result["extracted_explanation"] = extracted_explanation
        result["is_correct"] = is_correct
        result["indices_comparison"] = indices_comparison
        result["answer_time"] = answer_time
        result["finish_reason"] = finish_reason
        
    except Exception as e:
        print(f"Error processing question {question_id}: {str(e)}")
        result["error"] = str(e)
        result["raw_answer"] = f"Error: {str(e)}"
        result["extracted_answer"] = ""
        result["extracted_indices"] = []
        result["extracted_explanation"] = ""
        result["is_correct"] = False
        result["indices_comparison"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False
        }
        result["answer_time"] = 0
    
    return result

async def main():
    # Load the data
    with open("/root/trns-ai/data/train_multi_choice.json", "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from dataset")
    
    # Process each question
    results = []
    correct_count = 0
    exact_indices_count = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    
    for item in data:
        print(f"Processing question {item['id']}")
        result = await process_question(item)
        results.append(result)
        
        if result.get("is_correct", False):
            correct_count += 1
        print(f"Accuracy to the item {item['id']}: {correct_count}/{item['id']}")
        # Track indices metrics
        indices_comparison = result.get("indices_comparison", {})
        if indices_comparison.get("exact_match", False):
            exact_indices_count += 1
        
        precision_sum += indices_comparison.get("precision", 0)
        recall_sum += indices_comparison.get("recall", 0)
        f1_sum += indices_comparison.get("f1", 0)
            
        print(f"Completed question {item['id']} - Correct: {result.get('is_correct', False)}")
    
    # Calculate accuracy
    total_questions = len(data)
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    # Calculate indices metrics
    avg_precision = precision_sum / total_questions if total_questions > 0 else 0
    avg_recall = recall_sum / total_questions if total_questions > 0 else 0
    avg_f1 = f1_sum / total_questions if total_questions > 0 else 0
    indices_accuracy = exact_indices_count / total_questions if total_questions > 0 else 0
    
    # Prepare summary
    summary = {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "indices_metrics": {
            "exact_matches": exact_indices_count,
            "indices_accuracy": indices_accuracy,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Prepare final output
    output = {
        "summary": summary,
        "results": results
    }
    
    # Save results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/root/trns-ai/st_CoT/results/numerical_pipeline_results_{timestamp}.json"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Answer Accuracy: {accuracy:.2%} ({correct_count}/{total_questions})")
    print(f"Indices Accuracy: {indices_accuracy:.2%} ({exact_indices_count}/{total_questions})")
    print(f"Indices Avg F1: {avg_f1:.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 