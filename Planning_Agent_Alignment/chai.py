from datasets import load_dataset
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PPL ranker")

    parser.add_argument("--ds_plan", type=str, required=True)
    parser.add_argument("--ds_ground", type=str, required=True)
    parser.add_argument("--input_multi", type=str, required=True)
    parser.add_argument("--output_multi", type=str,default=5)

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    ds_plan = load_dataset('json', data_files=args.ds_plan, split='train')
    ds_ground = load_dataset('json', data_files=args.ds_ground, split='train')

    plan_ids = ds_plan['id']
    ground_ids = ds_ground['id']
    from tqdm import tqdm

    re_idx = []
    for idx in tqdm(range(len(plan_ids))):
        if plan_ids[idx] not in ground_ids:

            re_idx.append(idx)
    print(re_idx)
    df = pd.read_csv(args.input_multi)
    df = df.drop(index=re_idx).reset_index(drop=True)
    print(df.shape)
    print(len(ground_ids))
    # exit()
    df.to_csv(args.output_multi, index = False)
