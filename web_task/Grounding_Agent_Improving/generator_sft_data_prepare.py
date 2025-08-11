
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np

import random
from transformers import AutoTokenizer
from pathlib import Path

import argparse
import torch
from functools import partial
    
def encode_with_messages_format(example, tokenizer, max_seq_length):

    messages = example['messages']

    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
    
def format_row(example):
    re_messages = []
    messages = example['messages']

    for message in messages:
        re_messages.append({"role": message["role"], "content": message["content"]})
    
    return {'messages': re_messages}

def parse_args():
    
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input_file", default='', type=str)
    parser.add_argument("--input_file2", default='', type=str)
    parser.add_argument("--output_file", default='', type=str)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--max_new_tokens", default=512, type=int)

    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=1024,
        )
 

    ds2 = load_dataset('json', data_files=args.input_file2, split='train') 


    ds = load_dataset('json', data_files=args.input_file, split='train')


    messages = ds['co_messages']+ds2['messages']
    

    ds = Dataset.from_dict({'messages':messages})
    ds = ds.map(encode_function, remove_columns=ds.column_names, num_proc=8)
    ds.save_to_disk(args.output_file)