import argparse
import asyncio
from custom_symbcot import CUSTOM_MODEL

def parse_args():
    parser = argparse.ArgumentParser(description="Run chained question classification on train data")
    parser.add_argument('--data_path', type=str, default='../data/train.json', help='Path to the json file')
    parser.add_argument('--dataset_name', type=str, default='TRNS_AI', help='Name of the dataset to process')
    parser.add_argument('--save_path', type=str, default='./results', help='Path to save results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Name of the model to use')
    parser.add_argument('--stop_words', type=str, default='------', help='Stop words to use in generation')
    parser.add_argument('--mode', type=str, default='symbcot', help='Mode of operation')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Model temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Model top_p')
    parser.add_argument('--usePlan', action='store_true', help='Whether to use Plan Generation')
    parser.add_argument('--useStructured', action='store_true', help='Whether to use Structured Output')
    
    # Force ChainedQuestion type
    parser.add_argument('--question_type', type=str, default='ChainedQuestion', 
                       help='Type of question to process (fixed to ChainedQuestion for this script)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Processing {args.data_path} with question type: {args.question_type}")
    print(f"Using model: {args.model_name}")
    
    # Create model instance
    model = CUSTOM_MODEL(args)
    
    # Run the classification
    asyncio.run(model.reasoning_graph_generation()) 