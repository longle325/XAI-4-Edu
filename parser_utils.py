import re
from pydantic import BaseModel, Field
from typing import List, Literal
"""
This module contains utility functions for parsing the response from the solver.
"""

def parse_response_YesNo_question(response_c):
    """
    Parses the solver's response to extract the final answer, idx, and explanation.

    Args:
        response_c: The response from the solving stage.

    Returns:
        tuple: (final_answer, idx, explanation)
    """
    # Extract final answer
    pattern_answer = r"Final answer: \-{([AZa-z]+)\}"
    match_answer = re.search(pattern_answer, response_c)
    final_answer = match_answer.group(1).lower() if match_answer else "No final answer found in the text."

    # Extract idx
    pattern_idx = r"idx: \[([\d, ]+)\]"
    match_idx = re.search(pattern_idx, response_c)
    idx = [int(i.strip()) for i in match_idx.group(1).split(',')] if match_idx else []

    # Extract explanation
    pattern_explanation = r"explanation: (.+)"
    match_explanation = re.search(pattern_explanation, response_c)
    explanation = match_explanation.group(1).strip() if match_explanation else "No explanation provided."

    return final_answer, idx, explanation

def parse_verified_response_YesNo_question(response_c):
    """
    Parses the solver's response to extract the final answer, idx, and explanation.

    Args:
        response_c: The response from the solving stage.

    Returns:
        tuple: (final_answer, idx, explanation)
    """
    # Extract final answer
    pattern_answer = r"Final verified answer: \{([A-Za-z]+)\}"
    match_answer = re.search(pattern_answer, response_c)
    final_answer = match_answer.group(1).lower() if match_answer else "No final answer found in the text."

    # Extract idx
    pattern_idx = r"Verified idx: \[([\d, ]+)\]"
    match_idx = re.search(pattern_idx, response_c)
    idx = [int(i.strip()) for i in match_idx.group(1).split(',')] if match_idx else []

    # Extract explanation
    pattern_explanation = r"Verified explanation: (.+)"
    match_explanation = re.search(pattern_explanation, response_c)
    explanation = match_explanation.group(1).strip() if match_explanation else "No explanation provided."

    return final_answer, idx, explanation

def update_answer_YesNo(sample, plan, output, current_try, max_tries):
    """
    Creates a structured output dictionary including idx and explanation.

    Args:
        sample: Original example data (dict with 'id', 'question', 'context', 'answer')
        translation: Translation stage output (str)
        plan: Planning stage output (str)
        output: Solving stage output (str)

    Returns:
        dict: Structured output with results including idx and explanation
    """
    # Map solver answers to FOLIO labels
    
    # Parse solver response
    final_answer, idx, explanation = parse_response_YesNo_question(output)
    
    # Apply FOLIO-specific mapping

    idx.sort()
    dict_output = {
        'id': sample['id'],
        'question': sample['question'],
        'answer': final_answer,
        'true_answer': sample['answer'],
        'idx': idx,
        'explanation': explanation,
        'true_idx': sample.get('idx')
    }
    return dict_output

def construct_prompt_a_YesNo(record, in_context_examples_trans):
    """
    Constructs the prompt for the translation stage.

    Args:
    record: Data record containing context and question
    in_context_examples_trans: Translation examples text

    Returns:
    str: Complete prompt for translation
    """
    full_prompt = in_context_examples_trans
    context = record['context'].split(".")
    context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
    question = record['question'].strip()
    # Replace placeholders with actual content
    full_prompt = full_prompt.replace('[[CONTEXT]]', context)
    full_prompt = full_prompt.replace('[[QUESTION]]', question)
    return full_prompt

def construct_prompt_a_MultiChoice(record, in_context_examples_trans):
    """
    Constructs the prompt for the translation stage.

    Args:
    record: Data record containing context and question
    in_context_examples_trans: Translation examples text

    Returns:
    str: Complete prompt for translation
    """
    full_prompt = in_context_examples_trans
    context = record['context'].split(".")
    context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
    questionList = record['question'].strip().split("\n")
    question = questionList[0]
    conclusions = "\n".join(questionList[1:])
    # Replace placeholders with actual content
    full_prompt = full_prompt.replace('[[CONTEXT]]', context)
    full_prompt = full_prompt.replace('[[QUESTION]]', question)
    full_prompt = full_prompt.replace('[[OPTION]]', conclusions)
    # print(full_prompt)
    return full_prompt

def post_process_a_MultiChoice(response_a):
        """
        Extracts context and question from translation response.
        
        Args:
            response_a: Translation response to process
            
        Returns:
            tuple: (context, question) extracted from response
        """
        response_a = str(response_a)
        # Extract context using string indices
        context_end = response_a.find('Question:')
        context = response_a[0:context_end].strip()
        # Extract question using string indices
        question_start = response_a.find('Question:') + 9
        question_end = response_a[question_start:].find('Options:') + question_start
        question = response_a[question_start:question_end].strip()
        # Extract options using string indices
        options_start = response_a.find('Options:') + 8
        options_end = response_a[options_start:].find('###') + options_start
        options = response_a[options_start:options_end].strip()
        return context, question, options

def parse_response_MultiChoice(response_c):
    """
    Parses the solver's response to extract the final answer, idx, and explanation for multiple-choice questions.

    Args:
        response_c: The response from the solving stage.

    Returns:
        tuple: (final_answer, idx, explanation)
            - final_answer: str, the selected option (A, B, C, or D)
            - idx: list of int, the indices of premises used
            - explanation: str, the explanation text
    """
    # Handle non-string input
    if not isinstance(response_c, str):
        print(f"Error: Invalid response_c type: {type(response_c)}, value: {response_c}")
        return "Invalid response type", [], "No explanation provided."

    # Normalize whitespace and ensure string is not empty
    response_c = response_c.strip()
    if not response_c:
        print("Error: Empty response_c")
        return "No final answer found", [], "No explanation provided."

    # Extract final answer with flexible pattern
    pattern_answer = r'Final answer:\s*(?:\{(?:")?\s*([A-D])\s*(?:"})?|\s*([A-D])\s*)'
    match_answer = re.search(pattern_answer, response_c, re.IGNORECASE)
    final_answer = match_answer.group(1).upper() if match_answer and match_answer.group(1) else (match_answer.group(2).upper() if match_answer and match_answer.group(2) else "No final answer found")

    # Extract idx
    pattern_idx = r'idx:\s*\[([\d,\s]*)\]'
    match_idx = re.search(pattern_idx, response_c)
    idx = [int(i.strip()) for i in match_idx.group(1).split(',') if i.strip()] if match_idx else []

    # Extract explanation
    pattern_explanation = r'explanation:\s*(.+?)(?:\n|$|\Z)'
    match_explanation = re.search(pattern_explanation, response_c, re.DOTALL)
    explanation = match_explanation.group(1).strip() if match_explanation else "No explanation provided."

    return final_answer, idx, explanation

def update_answer_MultiChoice(sample, plan, output, current_try, max_retries):
    """
    Creates a structured output dictionary including idx and explanation for multiple-choice questions.

    Args:
        sample: Original example data (dict with 'id', 'question', 'context', 'answer', 'idx')
        translation: Translation stage output (str)
        plan: Planning stage output (str)
        output: Solving stage output (str)

    Returns:
        dict: Structured output with results including idx and explanation
    """
    # Parse solver response
    final_answer, idx, explanation = parse_response_MultiChoice(output)
    
    # Validate predicted answer
    predicted_answer = final_answer
    if predicted_answer not in ["A", "B", "C", "D"]:
        if current_try == max_retries:
            predicted_answer = "C"
        else: 
            return None

    # Validate required sample fields
    required_fields = ['id', 'question', 'context', 'answer']
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Sample missing required field: {field}")

    # Sort idx for consistency
    idx.sort()

    # Create structured output dictionary
    dict_output = {
        'id': sample['id'],
        'question': sample['question'],
        # 'original_context': sample['context'],
        # 'plan': plan,
        # 'execution': output,
        'predicted_answer': predicted_answer,
        'answer': sample['answer'],
        'idx': idx,
        'explanation': explanation,
        'true_idx': sample.get('idx', [])  # Use .get() to handle cases where 'idx' might be missing
    }
    return dict_output

def parse_response_ChainedQuestion(response_c):
    """
    Parses the solver's response to extract the final answer, idx, and explanation for multiple-choice questions.

    Args:
        response_c: The response from the solving stage.

    Returns:
        tuple: (final_answer, idx, explanation)
            - final_answer: str, the selected option (A, B, C, or D)
            - idx: list of int, the indices of premises used
            - explanation: str, the explanation text
    """
    # Handle non-string input
    if not isinstance(response_c, str):
        print(f"Error: Invalid response_c type: {type(response_c)}, value: {response_c}")
        return "Invalid response type", [], "No explanation provided."

    # Normalize whitespace and ensure string is not empty
    response_c = response_c.strip()
    if not response_c:
        print("Error: Empty response_c")
        return "No final answer found", [], "No explanation provided."

    # Extract final answer with flexible pattern
    pattern_answer = r'Final answer:\s*(\d*\.?\d+),\s*(\S.*)'
    match_answer = re.search(pattern_answer, response_c, re.IGNORECASE)
    final_answer = f"{match_answer.group(1)}, {match_answer.group(2)}".strip()
    print(final_answer)

    # Extract idx
    pattern_idx = r'idx:\s*\[([\d,\s]*)\]'
    match_idx = re.search(pattern_idx, response_c)
    idx = [int(i.strip()) for i in match_idx.group(1).split(',') if i.strip()] if match_idx else []

    # Extract explanation
    pattern_explanation = r'explanation:\s*(.+?)(?:\n|$|\Z)'
    match_explanation = re.search(pattern_explanation, response_c, re.DOTALL)
    explanation = match_explanation.group(1).strip() if match_explanation else "No explanation provided."

    return final_answer, idx, explanation

def update_answer_ChainedQuestion(sample, plan, output, current_try, max_retries):
    """
    Creates a structured output dictionary including idx and explanation for multiple-choice questions.

    Args:
        sample: Original example data (dict with 'id', 'question', 'context', 'answer', 'idx')
        translation: Translation stage output (str)
        plan: Planning stage output (str)
        output: Solving stage output (str)

    Returns:
        dict: Structured output with results including idx and explanation
    """
    # Parse solver response
    final_answer, idx, explanation = parse_response_ChainedQuestion(output)
    print(final_answer)
    print(idx)
    print(explanation)
    
    # Validate predicted answer
    predicted_answer = final_answer
    print(predicted_answer)
    if len(predicted_answer.split(", ")) != 2:
        if current_try == max_retries:
            predicted_answer = f"{predicted_answer}, No"
        else: 
            return None

    # Validate required sample fields
    required_fields = ['id', 'question', 'context', 'answer']
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Sample missing required field: {field}")

    # Sort idx for consistency
    idx.sort()

    # Create structured output dictionary
    dict_output = {
        'id': sample['id'],
        'question': sample['question'],
        # 'original_context': sample['context'],
        # 'plan': plan,
        # 'execution': output,
        'predicted_answer': predicted_answer,
        'answer': sample['answer'],
        'idx': idx,
        'explanation': explanation,
        'true_idx': sample.get('idx', [])  # Use .get() to handle cases where 'idx' might be missing
    }
    return dict_output

def parse_response_Numerical(response_c):
    """
    Parses the solver's response to extract the final answer, idx, and explanation for numerical questions.

    Args:
        response_c: The response from the solving stage.

    Returns:
        tuple: (final_answer, idx, explanation)
            - final_answer: str, the numerical answer
            - idx: list of int, the indices of premises used
            - explanation: str, the explanation text
    """
    # Handle non-string input
    if not isinstance(response_c, str):
        print(f"Error: Invalid response_c type: {type(response_c)}, value: {response_c}")
        return "Invalid response type", [], "No explanation provided."

    # Normalize whitespace and ensure string is not empty
    response_c = response_c.strip()
    if not response_c:
        print("Error: Empty response_c")
        return "No final answer found", [], "No explanation provided."

    # Extract final answer with flexible pattern
    pattern_answer = r'Final answer:\s*(.+)'
    match_answer = re.search(pattern_answer, response_c, re.IGNORECASE)
    final_answer = match_answer.group(1).strip() if match_answer else "No final answer found"

    # Extract idx
    pattern_idx = r'idx:\s*\[([\d,\s]*)\]'
    match_idx = re.search(pattern_idx, response_c)
    idx = [int(i.strip()) for i in match_idx.group(1).split(',') if i.strip()] if match_idx else []

    # Extract explanation
    pattern_explanation = r'explanation:\s*(.*)'
    match_explanation = re.search(pattern_explanation, response_c, re.DOTALL)
    explanation = match_explanation.group(1).strip() if match_explanation else "No explanation provided."

    return final_answer, idx, explanation

def update_answer_Numerical(sample, plan, output, current_try, max_retries):
    """
    Creates a structured output dictionary including idx and explanation for numerical questions.

    Args:
        sample: Original example data (dict with 'id', 'question', 'context', 'answer', 'idx')
        plan: Planning stage output (str)
        output: Solving stage output (str)
        current_try: Current attempt number (int)
        max_retries: Maximum allowed retries (int)

    Returns:
        dict: Structured output with results including idx and explanation, or None if retry needed
    """
    # Parse solver response
    final_answer, idx, explanation = parse_response_Numerical(output)

    # Validate predicted answer
    predicted_answer = final_answer
    # Check if answer is malformed (e.g., "No final answer found" or non-numerical)
    if predicted_answer == "No final answer found":
        if current_try == max_retries:
            predicted_answer = "0.0"  # Default fallback for numerical questions
        else:
            return None

    # Validate required sample fields
    required_fields = ['id', 'question', 'context', 'answer']
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Sample missing required field: {field}")

    # Sort idx for consistency
    idx.sort()

    # Create structured output dictionary
    dict_output = {
        'id': sample['id'],
        'question': sample['question'],
        # 'original_context': sample['context'],
        # 'plan': plan,
        # 'execution': output,
        'predicted_answer': predicted_answer,
        'answer': sample['answer'],
        'idx': idx,
        'explanation': explanation,
        'true_idx': sample.get('idx', [])  # Use .get() to handle cases where 'idx' might be missing
    }
    return dict_output


class YesNoResponse(BaseModel):
    Final_answer: Literal["Yes", "No", "Uncertain"] = Field(..., description='One of "Yes", "No", or "Uncertain", representing the conclusion based solely on the premises.')
    idx: List[int] = Field(..., description='List of indices (1-based) of the premises used to derive the final answer.')
    explanation: str = Field(..., description='A natural language explanation justifying the final answer based on the premises.')

class MCQResponse(BaseModel):
    Final_answer: Literal["A", "B", "C", "D"] = Field(..., description='The selected answer choice. Must be one of "A", "B", "C", or "D".')
    idx: List[int] = Field(..., description='List of indices (1-based) of the premises used to justify the chosen answer.')
    explanation: str = Field(..., description='A natural language explanation justifying the final answer based on the premises.')

class NumericalResponse(BaseModel):
    Final_answer: str = Field(
        ..., 
        description=(
            "The computed final answer to the question. "
            "This may include a single numeric value, a currency amount (e.g., with a '$' symbol), or a comma-separated list of values if multiple results are expected."
        ))
    idx: List[int] = Field(..., description='List of indices (1-based) of the premises used to justify the chosen answer.')
    explanation: str = Field(..., description='A natural language explanation justifying the final answer based on the premises.')

class ChainedResponse(BaseModel):
    Final_answer: str = Field(
        ..., 
        description=(
            "The computed final answer to the question. "
            "For these multi-part questions, this will be a string combining two values separated by a comma and a space: "
            "(1) the numeric result such as total credits or hours—an integer or a currency amount prefixed with a symbol (e.g., '$130'), "
            "and (2) a binary eligibility or decision flag ('Yes' or 'No')."
        ))
    idx: List[int] = Field(..., description='List of indices (1-based) of the premises used to justify the chosen answer.')
    explanation: str = Field(..., description='A natural language explanation justifying the final answer based on the premises.')