from datasets import load_dataset



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
from chat import get_from_openai
from prompt_reward import reward_prompt_web
import concurrent.futures

NUM_DUPS = 15

def format_row(example):
    re_messages = []
    messages = example['co_messages']
    re_messages.append({"role": "user", "content":messages[0]['content'].split('<|assistant|>\n')[0].replace('<|user|>\n','')})

    for me in messages[0]['content'].split('<|assistant|>\n'):
        if '\n\nTask:' in me or me.strip() == '':
            continue
        re_messages.append({"role": "assistant", "content":me.split('\n\n<|user|>\n')[0]})
        re_messages.append({"role": "user", "content":me.split('\n\n<|user|>\n')[1]})
    re_messages.append({"role": "assistant", "content":messages[1]['content']})

    return {'co_messages': re_messages}
    

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input_file", default='nq-train', type=str)
    parser.add_argument("--output_file", default='nq-train', type=str)
    
    
    
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    ds = load_dataset('json', data_files=args.input_file, split='train')

    def filter_function(example):
        return '=' in example['co_messages'][1]['content']

    # 过滤数据集
    ds = ds.filter(filter_function)



    ds = ds.map(format_row, num_proc=1, remove_columns=ds.column_names)
    
    ds.to_json(args.output_file)

