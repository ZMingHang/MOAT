from datasets import load_dataset, Dataset
import json
import argparse
from vllm import LLM, SamplingParams    
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"





NUM_SAMPLES = 15

plan_data_list=[]
def read_subgoal(file_path):
    global plan_data_list
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 将每一行解析为 JSON 对象，并添加到列表中
                json_data = json.loads(line.strip())
                plan_data_list.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}")
                print(f"Error: {e}")

def format_row(example):

    prompt_ground_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                                "The available action list is 'CLICK', 'TYPE', 'SELECT'.\nCLICK(Env, Query): Click the relevant html region in Env according to Query; TYPE(Env, Query, Text): Type Text into the relevant html region in Env according to Query; SELECT(Env, Query, Text): Select the value Text of the relevant selection box in Env according to Query.\n\n"

    
    assert plan_data_list[example['index_id']+int(example['id'].split('_')[-2])]['id'] == example['id']

    paras = plan_data_list[example['index_id']+int(example['id'].split('_')[-2])]['ctxs']
    

    example['input'] = []
    message_text = ''
    for message in example['messages'][:-2]:
        
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() +  "\n\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
        
    for idx in range(NUM_SAMPLES):
        # ground_prompt = "<|user|>\n" + prompt_ground_prefix + "Task: " + plan_data_list[example['index_id']+int(example['id'].split('_')[-2])].split('Please provide a reasonable subgoal-based plan to solve the given task.\nTask:')[-1].split(' Initial Environment Description: None.')[0].strip() + f" \nSubgoal to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"
        if message_text == '':
            example['input'].append("<|user|>\n" + example['messages'][0]['content'].split('Subgoal 1:')[0] + paras[idx]['subgoal'].strip() + '\n<|assistant|>\n')
        else:
            example['input'].append(message_text + f"<|user|>\nSubgoal to be grounded: {paras[idx]['subgoal'].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n")

    

    example['target'] = [example['messages'][-1]['content']]
    
    return example
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input_file", default='', type=str)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--subgoals_file", default='', type=str)
    parser.add_argument("--score_csv", default='', type=str)
    
    
    args = parser.parse_args()
    return args

def call_model_dup(prompts, target_outputs, model, max_new_tokens=50, num_samples=1):
                                                    
                                                    
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_samples))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_samples)])
    score_df = pd.read_csv(args.score_csv)
                                                    
    
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.95, max_tokens=max_new_tokens)
    
    odf = pd.DataFrame()
    input_list = []
    output_list = []
    score_list = []
    target_list = []
    for idx in range(num_samples):
        input_list.append(pdf[f'input_{idx}'].tolist())
        preds = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        preds = [pred.outputs[0].text for pred in preds]
        output_list.append(preds)
        score_list.append(score_df[f'output_{idx}'].tolist())
        target_list.append(target_outputs)
        
    return {
        'input': [item for sublist in input_list for item in sublist],
        'output':  [item for sublist in output_list for item in sublist]  ,
        'score': [item for sublist in score_list for item in sublist],
        'target_output': [item for sublist in target_list for item in sublist]
    }

from functools import partial                                                    
    
if __name__ == '__main__':
    args = parse_args()
    read_subgoal(args.subgoals_file)

    ds = load_dataset('json', data_files=args.input_file, split='train')


    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True)

    
    encode_function = partial(
            format_row,
        )


    ds = ds.map(encode_function, num_proc=8, remove_columns=ds.column_names)
    
    


    preds = call_model_dup(ds['input'],ds['target'], model, max_new_tokens=args.max_new_tokens, num_samples=NUM_SAMPLES)



    rds = Dataset.from_dict(preds)

    rds.to_json((args.output))
