import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
from vllm_utils import VLLM_MODEL

class VerificationResponse(BaseModel):
    verified: bool
    explanation: str
    answer: str = None  # Add answer field to track the verified answer

class Verifier:
    def __init__(self, vllm_client: VLLM_MODEL, symbcot_model):
        self.vllm_client = vllm_client
        self.symbcot_model = symbcot_model
        self.dataset_name = "TRNS_AI"
        self.temperature = 0.0
        self.max_retries = 3  # Maximum number of retries for verification

    def load_verification_prompt(self, question_type: str) -> str:
        file_path = os.path.join('../verification', self.dataset_name, question_type, 'verification.txt')
        with open(file_path) as f:
            return f.read()

    def construct_verification_prompt(self, context: str, question: str, answer: str, question_type: str) -> str:
        prompt_template = self.load_verification_prompt(question_type)
        prompt = prompt_template.replace('[[CONTEXT]]', context).replace('[[QUESTION]]', question).replace('[[ANSWER]]', answer)
        return prompt

    def standardize_yesno_answer(self, answer: str) -> str:
        """Standardize Yes/No answers to Yes, No, or Uncertain"""
        answer = answer.lower().strip()
        if answer in ['yes', 'true', 'correct']:
            return 'Yes'
        elif answer in ['no', 'false', 'incorrect']:
            return 'No'
        elif answer in ['uncertain', 'unknown', 'maybe']:
            return 'Uncertain'
        return answer  # Return original if not recognized

    async def verify(self, premises: List[str], question: str, answer: str, question_type: str) -> VerificationResponse:
        context = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises)])
        prompt = self.construct_verification_prompt(context, question, answer, question_type)
        response = await self.vllm_client.structured_output_chat_generate(prompt, temperature=self.temperature, output_structure=VerificationResponse)

        try:
            # Assume the model returns a JSON string with 'verified', 'explanation', and 'answer' fields
            result = response.model_dump()
            verified = result.get('verified', False)
            explanation = result.get('explanation', '')
            verified_answer = result.get('answer', answer)
            
            # Standardize Yes/No answers
            if question_type == "YesNo":
                verified_answer = self.standardize_yesno_answer(verified_answer)
            
            return VerificationResponse(
                verified=verified,
                explanation=explanation,
                answer=verified_answer
            )
        except (json.JSONDecodeError, ValidationError):
            return VerificationResponse(
                verified=False,
                explanation="Failed to parse verification response.",
                answer=answer
            )

    async def verify_and_plan(self, premises: List[str], question: str, answer: str, question_type: str) -> Dict[str, Any]:
        # Standardize the answer first if it's a Yes/No question
        if question_type == "YesNo":
            answer = self.standardize_yesno_answer(answer)
            
        # Try verification with original answer
        verification = await self.verify(premises, question, answer, question_type)
        
        if verification.verified:
            return {
                "verified": True,
                "explanation": verification.explanation,
                "answer": verification.answer,
                "idx": verification.get("idx", [])
            }
        
        # If verification fails, try alternative answers for Yes/No questions
        if question_type == "YesNo":
            alternative_answers = ["Yes", "No", "Uncertain"]
            for alt_answer in alternative_answers:
                if alt_answer != answer:  # Skip the original answer
                    verification = await self.verify(premises, question, alt_answer, question_type)
                    if verification.verified:
                        return {
                            "verified": True,
                            "explanation": verification.explanation,
                            "answer": verification.answer,
                            "idx": verification.get("idx", [])
                        }
        
        # If all verifications fail, replan and solve
        print("All verifications failed. Replanning...")
        result_dict = await self.symbcot_model.symbcot_reasoning_graph_generation(premises, question, question_type)
        
        # Standardize the answer if it's a Yes/No question
        if question_type == "YesNo" and "answer" in result_dict:
            result_dict["answer"] = self.standardize_yesno_answer(result_dict["answer"])
            
        return result_dict 