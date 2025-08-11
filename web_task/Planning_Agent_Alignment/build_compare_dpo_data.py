import pandas as pd
from datasets import load_dataset, Dataset
import argparse
import numpy as np
from pathlib import Path
import json


NUM_SAMPLES = 15

plan_data_list = []
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


    
def format_row(example):
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    assert plan_data_list[example['index_id']+int(example['id'].split('_')[-2])]['id'] == example['id']

    def _concat_messages(messages):
        message_text = ""
        for i in range(len(messages)) :
            message = messages[i]
            if i == len(messages) -1 :
                message_text += "<|assistant|>\n"
            elif message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() +  "\n\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text


    

    example_text = _concat_messages(plan_data_list[example['index_id']+int(example['id'].split('_')[-2])]['messages'])   

        
    
    return {'prompt': example_text}


    
def get_minmax(scores, threshold=0.4):
    assert scores.shape[1] == NUM_SAMPLES

    max_indices = np.argmax(scores, axis=-1)
    min_indices = np.argmin(scores, axis=-1)

    max_scores = scores[np.arange(scores.shape[0]), max_indices]
    min_scores = scores[np.arange(scores.shape[0]), min_indices]


    diff = max_scores - min_scores
    valid_mask = diff >= threshold

    max_indices = NUM_SAMPLES - 1 - max_indices
    min_indices = NUM_SAMPLES - 1 - min_indices

    return {
        "max": max_indices[valid_mask],
        "min": min_indices[valid_mask],
        "mask": valid_mask,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--input_score", type=str)
    parser.add_argument("--input_subgoals", type=str)
    parser.add_argument("--ds_ground", default='', type=str)
    parser.add_argument("--subgoals_file", default='', type=str)
    parser.add_argument("--output", required=True, type=str)
    
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    read_subgoal(args.subgoals_file)

    ds = load_dataset(
        'json', 
        data_files=args.ds_ground, 
        split='train'
    )

    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)

    sdf = pd.read_csv(args.input_score).values
    tdf = pd.read_csv(args.input_subgoals).values


    min_max = get_minmax(sdf, threshold=0.1)

    chosen = tdf[np.arange(tdf.shape[0])[min_max['mask']], min_max['min']].tolist()
    rejected = tdf[np.arange(tdf.shape[0])[min_max['mask']], min_max['max']].tolist()

    ds = ds.to_dict()

    valid_indices = np.where(min_max['mask'])[0]  
    filtered_ds_dict = {key: np.array(value)[valid_indices].tolist() for key, value in ds.items()}


    filtered_ds_dict['chosen'] = chosen
    filtered_ds_dict['rejected'] = rejected



    df = pd.DataFrame(filtered_ds_dict)
    filtered_df = df[df['rejected'].notna()]  
    filtered_ds_dict = filtered_df.to_dict(orient='list')
    ds = Dataset.from_dict(filtered_ds_dict)

    ds.to_json(args.output)
