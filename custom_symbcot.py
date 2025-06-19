import json                         
import os                           
from tqdm import tqdm               
from vllm_utils import VLLM_MODEL    
import argparse                                            
import asyncio                     
from parser_utils import *  

class CUSTOM_MODEL:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.question_type = self.args.question_type
        self.usePlan = self.args.usePlan
        self.use_structuredOutput = self.args.useStructured
        self.vllm_model = VLLM_MODEL(args.model_name, args.stop_words, args.max_new_tokens)
        self.model_name_safe = self.model_name.replace('/', '-')
        os.makedirs(self.save_path, exist_ok=True)

    def load_in_context_examples_plan(self):
        file_path = os.path.join('./prompts', self.dataset_name, self.question_type, 'plan_generation.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples


    def load_in_context_examples_solve(self):
        file_path = os.path.join('./prompts', self.dataset_name, self.question_type, 'solver.txt') if self.usePlan else os.path.join('./prompts', self.dataset_name, self.question_type, 'plan_generation.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self):
        with open(self.data_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def construct_prompt_b(self, context, question, in_context_examples_plan):
        full_prompt = in_context_examples_plan
        if self.question_type == "Numerical": 
            context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
        else: 
            context = context.split(".")
            context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        return full_prompt

    def construct_prompt_c(self, context, question, plan, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        if self.question_type == "Numerical": 
            context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
        else: 
            context = context.split(".")
            context = "\n".join([str(i+1) + "." + context[i] for i in range(len(context))])
        
        if self.usePlan: 
            full_prompt = full_prompt.replace('[[PLAN]]', plan)
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        return full_prompt


    async def procesOnequestion(self, example, pydanticTypeDict, parsedType_dict): 
        self.temperature = 0.0
        in_context_examples_plan = self.load_in_context_examples_plan()
        in_context_examples_solve = self.load_in_context_examples_solve()
        retries = 5

        print("Planning...")
        plan = await self.vllm_model.generate(self.construct_prompt_b(example["context"], example["question"], in_context_examples_plan), temperature=self.temperature, top_p=self.top_p) if self.usePlan else None
        dict_output = None
        print("Solving...")
        if self.use_structuredOutput:
            output = await self.vllm_model.generate(self.construct_prompt_c(example["context"], example["question"], plan , in_context_examples_solve), temperature=self.temperature, top_p=self.top_p, solve=True, response_format=pydanticTypeDict[self.question_type]) 
            dict_output = output.model_dump()
            dict_output["true_answer"] = example["answer"]
            dict_output["true_idx"] = example["idx"]
            dict_output["answer"] = dict_output["Final_answer"]
            dict_output["question"] = example["question"]
            del dict_output["Final_answer"]
        else: 
            output = await self.vllm_model.generate(self.construct_prompt_c(example["context"], example["question"], plan, in_context_examples_solve), temperature=self.temperature, top_p=self.top_p)
            update_answer = parsedType_dict[self.question_type]
            for i in range(1, retries+1): 
                dict_output = update_answer(example, plan, output, i , retries)
                if dict_output: 
                    break
                print(f"Unexpected output. Generating again: {i} ...")
                self.temperature = 0.2
                output = await self.vllm_model.generate(self.construct_prompt_c(example["context"],example["question"], plan, in_context_examples_solve), temperature=self.temperature,top_p=self.top_p)
        return dict_output

    async def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset()
        print("Loaded the dataset successfully!")
        pydanticTypeDictDict = {
            "YesNo": YesNoResponse,
            "MultiChoice": MCQResponse,
            "Numerical" : NumericalResponse,
            "ChainedQuestion": ChainedResponse
        } 

        parsedType_dict = {
            "YesNo": update_answer_YesNo,
            "MultiChoice": update_answer_MultiChoice,
            "Numerical" : update_answer_Numerical,
            "ChainedQuestion": update_answer_ChainedQuestion
        }
        outputs = []
        for example in tqdm(raw_dataset):
            try:
                dict_output = await self.procesOnequestion(example, pydanticTypeDictDict, parsedType_dict)
                outputs.append(dict_output)
                with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name_safe}_{self.question_type}.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print("Error in generation: ", e)

        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name_safe}_{self.question_type}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset and generate model responses")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the json file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--save_path', type=str, default='./results', help='Path to save results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Name of the model to use')
    parser.add_argument('--stop_words', type=str, default='------', help='Stop words to use in generation')
    parser.add_argument('--mode', type=str, default='symbcot', help='Mode of operation')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Model temperature.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Model top_p.')

    parser.add_argument('--usePlan', action='store_true', help='Whether to use Plan Generation.')
    parser.add_argument('--useStructured', action='store_true', help='Whether to use Structured Output.')

    parser.add_argument("--question_type", type=str, choices=["YesNo", "MultiChoice", "OpenEnded", "ChainedQuestion", "Numerical", "None"], default="None", help="Type of question to process")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    vllm_model = CUSTOM_MODEL(args)
    asyncio.run(vllm_model.reasoning_graph_generation())
