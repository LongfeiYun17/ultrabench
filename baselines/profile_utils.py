import jsonlines
import sys
from dataclasses import asdict
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
from openai import OpenAI
from google import genai
from google.genai import types
import os
import json
import collections
import time
import anthropic

sys.path.append(".")
from eval.ifeval.run_eval import (
    InputExample, test_instruction_following_strict
)

global_client = None

def init_worker(model_name):
    global global_client
    if 'gpt' in model_name:
        global_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif 'claude' in model_name:
        global_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    elif 'gemini' in model_name:
        global_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_data(data_path):
    data = []
    with jsonlines.open(data_path, "r") as reader:
        for line in reader:
            hard_constraints = line["hard_constraints"]
            soft_constraints = line["soft_constraints"]
            data.append({
                "key": line["key"],
                "hard_constraints": hard_constraints,
                "soft_constraints": soft_constraints,
                "instruction":  "You are a helpful assistant. Please generate a response that follows the instructions below.\n" \
                    + '\n'.join(soft_constraints["prompt"]) + '\n' + '\n'.join(hard_constraints["prompt"])
            })
    return data

def evaluate_single(item, eval_model, verify_template):
    global global_client
    inp = InputExample(key=item['key'],
                instruction_id_list=item['hard_constraints']['instruction_id_list'],
                prompt=item['instruction'],
                kwargs=item['hard_constraints']['kwargs'])
    prompt_to_response = {}
    prompt_to_response[item['instruction']] = item['response']
    res = test_instruction_following_strict(inp, prompt_to_response)
    item['hard_constraints']['follow_all_instructions'] = res.follow_all_instructions
    item['hard_constraints']['follow_instruction_list'] = res.follow_instruction_list

    text = item['response']
    soft_constraints = item['soft_constraints']['prompt']
    number = len(soft_constraints)
    prompt = verify_template.format(number=number, text=text, constraints='\n'.join(soft_constraints))
    retry_count, max_retries = 0, 3
    while retry_count < max_retries:
        try:
            response = global_client.responses.create(
                model=eval_model,
                input=prompt,
                temperature=0.1
            )
            output = response.output[0].content[0].text.strip().split("\n")
            output = [x.strip() for x in output]
            assert len(output) == number
            output = [True if x.lower() == 'yes' else False for x in output]
            item['soft_constraints']['follow_instruction_list'] = output
            break

        except Exception as e:
            retry_count += 1
            time.sleep(0.5)
            if retry_count == max_retries:
                print(f"Error: {e}")
                item['soft_constraints']['follow_instruction_list'] = [False for _ in range(number)]
    return item

def evaluate_outputs(outputs, eval_model, verify_template, single_process=True, num_processes=16):
    global global_client
    results = []
    if single_process:
        init_worker(eval_model)
        for item in tqdm(outputs):
            res = evaluate_single(item, eval_model, verify_template)
            results.append(res)
    else:
        with Pool(processes=num_processes, initializer=init_worker, initargs=(eval_model,)) as pool:
            fn = partial(evaluate_single, eval_model=eval_model, verify_template=verify_template)
            for res in tqdm(pool.imap(fn, outputs), total=len(outputs)):
                results.append(res)
        pool.close()
    return results

def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example['hard_constraints']['follow_instruction_list']
        instruction_id_list = example['hard_constraints']['instruction_id_list']
        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1
            
    metrics = {
        "prompt-leval accuracy": prompt_correct / prompt_total,
        "instruction-level accuracy": instruction_correct / instruction_total,
        "tier0 accuracy": {instruction_id: tier0_correct[instruction_id] / tier0_total[instruction_id] for instruction_id in tier0_total},
        "tier1 accuracy": {instruction_id: tier1_correct[instruction_id] / tier1_total[instruction_id] for instruction_id in tier1_total},
    }

    print(json.dumps(metrics, indent=4))
    return metrics
 
def compute_metrics(outputs):
    follow_all_instructions = [o['hard_constraints']['follow_all_instructions'] for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    results = {'Hard_Constraints': {}, 'Soft_Constraints': {}}
    results["Hard_Constraints"]["Accuracy"] = accuracy
 
    detailed_scores = print_report(outputs)
    results["Hard_Constraints"].update(detailed_scores)

    soft_scores = []
    for o in outputs:
        soft_constraints = o['soft_constraints']['follow_instruction_list']
        soft_scores.append(sum(soft_constraints) / len(soft_constraints))
    results["Soft_Constraints"]["Accuracy"] = sum(soft_scores) / len(soft_scores)
    return results

def generate_single_gpt(item, model_name, top_p, temperature, max_new_tokens):
    retry_count, max_retries = 0, 3
    while retry_count < max_retries:
        try:
            response = global_client.responses.create(
                model=model_name,
                input=item["instruction"],
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_new_tokens
            )
            output = response.output[0].content[0].text
            return {
                **item,
                "response": output
            }
        except Exception as e:
            retry_count += 1
            time.sleep(1)
            if retry_count == max_retries:
                print(f"Error: {e}")
                return {
                    **item,
                    "response": ""
                }
            
def generate_single_gemini(item, model_name, top_p, temperature, max_new_tokens):
    retry_count, max_retries = 0, 5
    while retry_count < max_retries:
        try:
            response = global_client.models.generate_content(
                model=model_name,
                contents=item["instruction"],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE
                        )
                    ]
                ),
            )
            output = response.candidates[0].content.parts[0].text
            return {
                **item,
                "response": output
            }
        except Exception as e:
            retry_count += 1
            time.sleep(0.5)
            if retry_count == max_retries:
                print(f"Error: {e}")
                return {
                    **item,
                    "response": ""
                }

def generate_single_claude(item, model_name, top_p, temperature, max_new_tokens):
    retry_count, max_retries = 0, 3
    while retry_count < max_retries:
        try:
            response = global_client.messages.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": item["instruction"]}
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            output = response.content[0].text
            return {
                **item,
                "response": output
            }   
        except Exception as e:
            retry_count += 1
            time.sleep(0.5)
            if retry_count == max_retries:
                print(f"Error: {e}")
                return {
                    **item,
                    "response": ""
                }
            
def generate_response(args, data):
    global global_client
    outputs = []
    if 'gpt' in args.model_name:
        if args.single_process:
            init_worker(args.model_name)
            for item in tqdm(data):
                res = generate_single_gpt(item, args.model_name, args.top_p, args.temperature, args.max_new_tokens)
                outputs.append(res)
        else:
            with Pool(processes=args.num_processes, initializer=init_worker, initargs=(args.model_name,)) as pool:
                fn = partial(generate_single_gpt, model_name=args.model_name, top_p=args.top_p, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                for res in tqdm(pool.imap(fn, data), total=len(data)):
                    outputs.append(res)
            pool.close()
    elif 'claude' in args.model_name:
        if args.single_process:
            init_worker(args.model_name)
            for item in tqdm(data):
                res = generate_single_claude(item, args.model_name, args.top_p, args.temperature, args.max_new_tokens)
                outputs.append(res)
        else: 
            with Pool(processes=args.num_processes, initializer=init_worker, initargs=(args.model_name,)) as pool:
                fn = partial(generate_single_claude, model_name=args.model_name, top_p=args.top_p, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                for res in tqdm(pool.imap(fn, data), total=len(data)):
                    outputs.append(res)
            pool.close()
    elif 'gemini' in args.model_name:
        if args.single_process:
            init_worker(args.model_name)
            for item in tqdm(data):
                res = generate_single_gemini(item, args.model_name, args.top_p, args.temperature, args.max_new_tokens)
                outputs.append(res)
        else:
            with Pool(processes=args.num_processes, initializer=init_worker, initargs=(args.model_name,)) as pool:
                fn = partial(generate_single_gemini, model_name=args.model_name, top_p=args.top_p, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                for res in tqdm(pool.imap(fn, data), total=len(data)):
                    outputs.append(res)
            pool.close()
    else:
        raise ValueError(f"Model {args.model_name} is not supported")
    return outputs