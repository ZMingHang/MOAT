
import logging
import sys
import os
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
import pickle

from modeling_ppllama import LlamaPPL
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
    DataCollatorForSeq2Seq, 
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torch.utils.data import Subset
import re
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
from tqdm.std import *
import numpy as np
import json

plan_data_list = []
def template_from_file(example):
    global plan_data_list
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
            example['input'].append(message_text + f"\n\n<|user|>\nSubgoal to be grounded: {paras[idx]['subgoal'].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n")

    example['target'] = [example['messages'][-1]['content'] for _ in range(NUM_SAMPLES)]
    
    return example

    
def format_tokenize_row(example, tokenizer):

    assert tokenizer.padding_side == 'left'
    input_ = example['input'][0]
    target = example['target'][0]

    encs = tokenizer(input_, padding=True, add_special_tokens=False)
    example['input_ids'] = encs['input_ids']
    example['attention_mask'] = encs['attention_mask']
    
    ans_encs = tokenizer(target, add_special_tokens=False)
    
    example['labels'] = [[-100] * len(row_enc) for row_enc in example['input_ids']]
    

    for idx, item in enumerate(example['labels']):
        example['input_ids'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['labels'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['attention_mask'][idx] += [1] * len(ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        assert len(example['input_ids'][idx]) == len(example['labels'][idx])
        assert len(example['attention_mask'][idx]) == len(example['labels'][idx])
    return example

def parse_args():
    parser = argparse.ArgumentParser(description="PPL ranker")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--num_samples", type=int,default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subgoal_file", type=str, required=True)

    
    args = parser.parse_args()
    return args
def read_subgoal(file_path):
    global plan_data_list
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_data = json.loads(line.strip())
                plan_data_list.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}")
                print(f"Error: {e}")

def main():
    args = parse_args()
    read_subgoal(args.subgoal_file)
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaPPL.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ds = load_dataset('json', data_files=args.input_file, split='train')
    
    with accelerator.main_process_first():
        ds = ds.map(template_from_file, num_proc=8, remove_columns=ds.column_names)
        ds = ds.map(format_tokenize_row, fn_kwargs={'tokenizer': tokenizer}, num_proc=8, remove_columns=ds.column_names, batched=True, batch_size=1)



    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_outdir",
        save_strategy = "no",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        eval_dataset=ds
    )

    preds = trainer.predict(ds)
    accelerator.wait_for_everyone()
    
    preds = preds.predictions[:ds.num_rows].reshape((-1, args.num_samples))
    odf = pd.DataFrame(preds, columns=[f'output_{idx}' for idx in range(args.num_samples)])
    odf.to_csv(args.output, index=False)

if __name__ == "__main__":

    NUM_SAMPLES = 15
    main()
