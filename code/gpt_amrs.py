import os
import json
from efficiency.nlp import Chatbot
from efficiency.log import fread
from efficiency.function import set_seed

import random
import numpy as np
import re
import ipdb
from tqdm import tqdm
import pandas as pd
import itertools
from pathlib import Path
from sklearn.utils import shuffle
random.seed(0)
set_seed(0)



model_version = 'gpt-3.5-turbo-0613'


root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"
current_dir = Path(__file__).parent.resolve()
gpt_outputs_dir = data_dir / "outputs"
parsed_dir  = data_dir / "parsed_amrs"



def query_amrs(input_file = data_dir / 'amr-release-test-bio-v0.8.csv', out_file = data_dir / f'outputs/{model_version}_bio_amr_test.csv', org_id="OPENAI_ORG_ID", model_version = 'gpt4-0613'):
    save_path = gpt_outputs_dir
    system_prompt = "You are a linguist of English with expertise in abstract meaning representation (AMR)."
    chat = Chatbot(model_version=model_version,max_tokens = 2048,
                   output_file=f'{save_path}/.cache_{model_version}_responses.csv',
                   system_prompt="system_prompt",
                   openai_key_alias='OPENAI_API_KEY',
                   openai_org_alias=org_id,
                   )
    chat.clear_dialog_history()
    with open(current_dir/'code/prompt_amr.txt') as f:
        template = f.read().strip()
        print(template)

    read_in_file = input_file
    local_out_file = out_file

    df = pd.read_csv(read_in_file)

    print("Now processing ", len(df), "data in test set")
    for index, row in tqdm(df.iterrows(), total = len(df)):
        print("Now processing ", index, row['id'])
        raw_prompt = template.format(**{"premise": row['text_detok'].strip()})
        pred = chat.ask(raw_prompt, system_prompt=system_prompt, stop_sign="########################")
        new_dict = {'raw_prompt': raw_prompt,
                    f'{model_version}_amr': pred}
        print(raw_prompt.split('\n')[-1])
        for key, val in new_dict.items():
            df.loc[index, key] = val

        df.to_csv(local_out_file, index = False)
    df.to_csv(local_out_file, index = False)
    print("Finished processing ", len(df), "data in test set")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--org', type=str,
                      default= "OPENAI_ORG_ID",
                      help='OpenAI organization ID')
    parser.add_argument('--model_version', type=str,default= 'gpt4-0613',help = 'model version')
    parser.add_argument('--input_file', type=str,default= data_dir / 'amr-release-test-bio-v0.8.csv',help = 'input file')
    parser.add_argument('--out_file', type=str,default= data_dir / f'outputs/{model_version}_bio_test_amr.csv',help = 'output file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args.org)
    query_amrs(args.org, args.model_version)

if __name__ == '__main__':
    main()

