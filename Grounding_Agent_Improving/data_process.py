from datasets import load_dataset, Dataset
import json
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import re
from prompt_reward import reward_prompt_math
import concurrent.futures
from vllm import LLM, SamplingParams    
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input_file", default='', type=str)
    parser.add_argument("--output_file", default='', type=str)
    
    
    
    args = parser.parse_args()
    return args

NUM_SAMPLES = 10

sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
# model = LLM(model="/DeepSeek-R1-Distill-Qwen-32B", tensor_parallel_size=1, trust_remote_code=True)
model = LLM(model="/data/szl/DeepSeek-R1-Distill-Qwen-32B/", tensor_parallel_size=1, trust_remote_code=True,max_model_len=5012)


if __name__ == '__main__':
    args = parse_args()
    ds = load_dataset('json', data_files = args.input_file, split='train')
    df = ds.to_pandas()

    messages = []
    co_messages = []
    num = 0
    rewards = []

    prompts = []
    print(int(df.shape[0]/NUM_SAMPLES))

    for i in tqdm(range( int(df.shape[0]/NUM_SAMPLES))):
        score_list = [df['score'].iloc[i],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*1],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*2],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*3],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*4]]
        max_index = score_list.index(min(score_list))

        message = []
        co_message = []
        message.append({"role": "user", "content": df['input'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]})
        message.append({"role": "assistant", "content": df['output'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]})
        
        task = df['input'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index].split('\n\nTask: ')[-1].split('\n')[0]
        subgoals = df['input'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index].replace('\n\n\n\n','\n\n').split(task)[-1].replace('\n<|assistant|>\n',' ').replace('\n\n<|user|>\n',' ')

        actions = df['output'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]



        
        prompt = reward_prompt_math.format(TASK = task, SUBGOALS = subgoals, ACTIONS = actions)
        prompts.append(prompt)

    preds = model.generate(prompts, sampling_params)
    results = [pred.outputs[0].text for pred in preds]


    for i in tqdm(range(  int(df.shape[0]/NUM_SAMPLES))):
        result = results[i]
        score_list = [df['score'].iloc[i],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*1],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*2],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*3],df['score'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*4]]
        max_index = score_list.index(min(score_list))

        message = []
        co_message = []
        message.append({"role": "user", "content": df['input'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]})
        message.append({"role": "assistant", "content": df['output'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]})


        if 'final answer reached' in result.lower():
            rewards.append(1)
            co_messages.append(message)
            messages.append(message)
        else:
            rewards.append(0)
            co_message.append({"role": "user", "content": df['input'].iloc[i+int(df.shape[0]/NUM_SAMPLES)*max_index]})
            try:


                match = re.search(r"Corrected Subgoals:\s*([^\n]+)", result)
                co_message.append({"role": "plan", "content": match.group(1).strip()})
                match = re.search(r"Corrected Actions:\s*([^\n]+)", result)
                co_message.append({"role": "action", "content": match.group(1).strip()})
                co_messages.append(co_message)
                messages.append(message)
            except:

                co_messages.append(message)
                messages.append(message)

    ds = Dataset.from_dict({'messages':messages,'reward': rewards,'co_messages':co_messages})
    ds.to_json(args.output_file)
