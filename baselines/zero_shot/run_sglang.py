import json
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
import sglang as sgl
import sys
sys.path.append(str(Path(__file__).parent.parent))
from profile_utils import load_data, evaluate_outputs, compute_metrics
from utils.api import api_price

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--model_name", type=str, default="qwen/qwen2.5-0.5b-instruct")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/zero_shot")
    parser.add_argument("--single_process", action="store_true")
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--test_data", type=str, default="data/eval/ifeval/fineweb_200000_test.jsonl")
    parser.add_argument("--eval_model", type=str, default="gpt-4o")
    parser.add_argument("--verify_template", type=str, default="prompt/verify.md")

    args = parser.parse_args()

    data = load_data(args.test_data)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B') 
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": item["instruction"]}], tokenize=False, add_generation_prompt=True) for item in data]
    llm = sgl.Engine(model_path=args.model_name, log_level="info")
    sampling_params = {
        "temperature": args.temperature, 
        "top_p": args.top_p, 
        "repetition_penalty": args.repetition_penalty, 
        "max_new_tokens": args.max_new_tokens
    }

    responses = llm.generate(prompts, sampling_params)

    llm.shutdown()
    
    outputs = [
        {
            **item,
            "response": response['text']
        } for item, response in zip(data, responses)
    ]

    with open(args.verify_template, 'r') as f:
        verify_template = f.read()
    eval_results = evaluate_outputs(outputs, args.eval_model, verify_template, single_process=args.single_process, num_processes=args.num_processes)

    eval_id = f'{args.model_name.split("/")[-1]}_max_new_tokens_{args.max_new_tokens}_temperature_{args.temperature}_zero_shot'
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f'{eval_id}.json'

    with open(output_path, 'w') as f:
        json.dump(eval_results, f)

    
    metrics = compute_metrics(eval_results)
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f'{eval_id}_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()