from datasets import load_dataset, Dataset
import json
import argparse
from vllm import LLM, SamplingParams    
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer




NUM_SAMPLES = 15
def format_row(example, tokenizer):
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

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

    prompts = []
    for i in range(NUM_SAMPLES):
        example_text = _concat_messages(messages) 

        prompts.append(example_text)
    
    return {'prompt': prompts}
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--output_name", default='', type=str)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--world_size", default=4, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    parser.add_argument("--ds_plan", required=True, type=str)
    
    
    args = parser.parse_args()
    return args

def call_model_dup(prompts, model, max_new_tokens=50, num_samples=1):
                                                    
                                                    
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_samples))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_samples)])

                                                    
    
    sampling_params = SamplingParams(
        temperature=1, top_p=0.95, max_tokens=max_new_tokens,stop = ['\n'])
    
    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_samples)])
    for idx in range(num_samples):

        preds = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        preds = [pred.outputs[0].text for pred in preds]
        odf[f'output_{idx}'] = preds                                            
    return odf

from functools import partial                                                    
    
if __name__ == '__main__':
    args = parse_args()
    output_name = args.output_name
    ds = load_dataset('json', data_files=args.ds_plan, split='train')


    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encode_function = partial(
            format_row,
            tokenizer=tokenizer
        )

    ds = ds.map(encode_function, num_proc=8, remove_columns=ds.column_names)
    
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_samples=NUM_SAMPLES)

    dest_dir = Path(args.dest_dir)
    if not dest_dir.exists():
        dest_dir.mkdir()

    model_name = Path(args.model_name).name

    preds.to_csv((dest_dir / f'{output_name}').resolve(), index=False)