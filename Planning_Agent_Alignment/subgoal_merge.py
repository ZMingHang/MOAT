from datasets import load_dataset, Dataset
from pathlib import Path
import pandas as pd
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--output_name", default='web_agent', type=str)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--subgoal_file", default='', type=str)
    parser.add_argument("--ds_source", default='', type=str)
    parser.add_argument("--dest_dir", default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    output_name = args.output_name

    subgoal_file_path = args.subgoal_file
    ds_source_path = args.ds_source

    dest_path = Path(args.dest_dir)
    if not dest_path.exists():
        dest_path.mkdir()


    subgoal_df = pd.read_csv(subgoal_file_path)
    subgoal_df = subgoal_df.fillna(subgoal_df['output_0'].iloc[0])

    rds = load_dataset('json', data_files=ds_source_path, split='train')
    
    nads = []
    for idx in range(args.num_samples):
        outputs = subgoal_df[f'output_{idx}'].tolist()
        ads = Dataset.from_dict({'output': outputs})
        nads.append(ads)
    
    rds = rds.to_list()


    for idx, item in enumerate(rds):
        rds[idx]['ctxs'] = []
        for jdx in range(args.num_samples):
            insert_item = {
                "id": f"output_{jdx}",
                "subgoal": nads[jdx][idx]['output'],
            }
            rds[idx]['ctxs'].insert(0, insert_item)

    rds = Dataset.from_list(rds)

    rds.to_json((dest_path / f'{output_name}').resolve())