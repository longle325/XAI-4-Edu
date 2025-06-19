import os                           
import re
from vllm_utils import Response

class SymbCoT:
    """
    A custom model class that implements a multi-step reasoning approach using language models.
    It processes datasets through three stages: translation, planning, and solving.
    """
    def __init__(self, client):
        self.question_type = None 
        self.dataset_name = "TRNS_AI" 
        self.temperature = 0.0
        self.vllm_client = client
        self.model_name_safe = "Qwen/Qwen3-32B-AWQ"
    
    def load_classification_prompt(self, question):
        file_path = '../prompts/TRNS_AI/classify_question.txt'
        with open(file_path) as f:
            classification_prompt = f.read()

        full_classification_prompt = classification_prompt.replace("[[QUESTION]]", question)

        return full_classification_prompt

    def load_in_context_ultimate_prompt(self):
        file_path = os.path.join('../prompts', self.dataset_name, 'ultimate_prompt.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_examples_plan(self):
        file_path = os.path.join('../prompts', self.dataset_name, self.question_type, 'plan_generation.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_extract(self):
        file_path = os.path.join('../prompts', self.dataset_name, self.question_type, 'extract.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def construct_prompt_for_answer(self, context, in_context_examples_plan, question='',question_type=None):
        full_prompt = in_context_examples_plan
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        return full_prompt

    def construct_prompt_for_extract(self, input, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        full_prompt = full_prompt.replace('[[INPUT]]', input)

        return full_prompt

    async def question_classifier(self, question):
        try:
            classification_prompt = self.load_classification_prompt(question)
            classification_answer = await self.vllm_client.structured_output_generate(classification_prompt, temperature=0.0, output_structure=Response)

            classification_answer_dict = classification_answer.model_dump()

            question_type = classification_answer_dict["question_type"]
            return question_type
        except Exception as e:
            print("Error while trying to classify question: ", e)
            return None
    
    async def symbcot_reasoning_graph_generation(self, premises, question, question_type):
        self.question_type = question_type
        self.temperature = 0.0
        retries = 5
        in_context_ultimate_prompt = self.load_in_context_ultimate_prompt()

        context = "\n".join([str(i+1) + "." + premises[i] for i in range(len(premises))])
        print("Context:", context)

        print("Answering question...")
        prompts_for_answer = self.construct_prompt_for_answer(context, in_context_ultimate_prompt, question, self.question_type)
        for i in range(retries):
            try:
                answer = await self.vllm_client.structured_output_generate(prompts_for_answer, temperature=self.temperature, output_structure=Response)
                break
            except:
                print(f"Error while trying to generate structured output: Generating again... {i+1}/{retries}")
                self.temperature += 0.05  
        
        tail_length = 50 
        tail = answer["explanation"][-tail_length:]
        pattern = rf"\${re.escape(str(answer['Final_answer']))}[\.\,\?\!\s]*$"

        if re.search(pattern, tail):
            answer["Final_answer"] = f"${answer['Final_answer']}" 
        return answer

def initialize_symbcot_model(client):
    try:
        symbcot_model = SymbCoT(client)
        return symbcot_model

    except Exception as e:
        print(f"Error while init the vllm model in the initialize_symbcot_model function: {e}" )
        return None
