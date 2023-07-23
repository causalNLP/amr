import pandas as pd
from pathlib import Path
import os
import random
import re

curret_dir = Path(__file__).parent.resolve()
data_dir = Path(__file__).parent.resolve() / "data"
parsed_dir  = data_dir / "parsed_amrs"

def extract_amr(s):
    match = re.search(r'\. \((.*)\)', s)
    if match:
        return "(" + match.group(1) + ")"

    match = re.search(r'\: \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    match = re.search(r'\" \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    match = re.search(r' \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    else:
      # print("No match in ", s)
      return None


def replace_sentence(s):
    if s is None:
        return None
    pattern = r"The abstract meaning representation.*as follows:"
    return re.sub(pattern, '', s).strip()

def parse_string(s):
    count_open = 0
    count_close = 0
    if s is None:
        return None
    try:
        s = s.replace("[", "(").replace("]", ")")
        s = re.sub("~\d+", "", s)
        s = re.sub("\t", " ", s)
        s = re.sub("\s+", " ", s)
    except Exception as e:
        return None

    for i in range(len(s)-1, -1, -1):
        if s[i] == ')':
            count_close += 1
        elif s[i] == '(':
            count_open += 1
        if count_open == count_close:
            return s[i:]
    return s




def parse_string_simple(s):
    def escape_slash_not_surrounded_by_spaces(text):
        return re.sub(r'(?<! )/(?! )', r'\/', text)
    s = escape_slash_not_surrounded_by_spaces(s)
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub("~\d+", "", s)
    s = re.sub("\t", " ", s)
    s = re.sub("\s+", " ", s)
    return s


def balance_parentheses(s):
    if s is None:
        return None
    stack = []
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                stack.pop()
            else:
                s = s[:i] + s[i+1:]
                return balance_parentheses(s)
    while stack:
        i = stack.pop()
        s = s[:i] + s[i+1:]
    return s


def merge_gpt_amrs():
    df_all = pd.read_csv(parsed_dir / "all_amrs.csv")
    df_gpt4_bio = pd.read_csv(parsed_dir / "gpt4-0613_bio_amr_test.csv")
    df_gpt3_bio = pd.read_csv(parsed_dir / "gpt-3.5-turbo-0613_bio_amr_test.csv")
    df_gpt4_ldc = pd.read_csv(parsed_dir / "gpt4-0613_ldc_amr_2.0.csv")
    df_gpt3_ldc = pd.read_csv(parsed_dir / "gpt-3.5-turbo-0613_ldc_amr_2.0.csv")

    # Replace "-" with "_" in column names
    df_gpt4_bio.columns = df_gpt4_bio.columns.str.replace("-", "_")
    df_gpt3_bio.columns = df_gpt3_bio.columns.str.replace("-", "_")
    df_gpt4_ldc.columns = df_gpt4_ldc.columns.str.replace("-", "_")
    df_gpt3_ldc.columns = df_gpt3_ldc.columns.str.replace("-", "_")


    df_gpt3 = pd.concat([df_gpt3_bio, df_gpt3_ldc], axis=0)
    df_gpt4 = pd.concat([df_gpt4_bio, df_gpt4_ldc], axis=0)

    df_all = df_all.merge(df_gpt3[['id','gpt_3.5_turbo_0613_amr']], on='id', how='left')
    print(df_all.columns)
    df_all = df_all.merge(df_gpt4[['id','gpt4_0613_amr']], on='id', how='left')
    print(df_all.columns)
    df_all.to_csv(parsed_dir / "all_amrs.csv", index=False)
    print(df_all.shape)



def main():
    df_all = pd.DataFrame()
    for file in os.listdir(parsed_dir):
        if file.endswith(".csv") and not file.startswith("all") or file.startswith("gpt"):
            df = pd.read_csv(parsed_dir / file)
            print(file, "with shape", df.shape)
            model = file.replace("amrs.csv", "").replace("_amr.csv", "")
            df.columns = ['id','text',f'{model}_amr']
            if 'bio' in file:
                if 'train' in file:
                    gold_df = pd.read_csv(data_dir / "amr-release-training-bio-v0.8.csv")
                    df['split'] = 'train'
                elif 'dev' in file:
                    gold_df = pd.read_csv(data_dir / "amr-release-dev-bio-v0.8.csv")
                    df['split'] = 'dev'
                elif 'test' in file:
                    gold_df = pd.read_csv(data_dir / "amr-release-test-bio-v0.8.csv")
                    df['split'] = 'test'
                df['source_set'] ='bio_amr0.8'
            elif 'ldc' in file:
                gold_df = pd.read_csv(data_dir / "ldc_gold_amrs_clean.csv")
                df['split'] = df['id'].apply(lambda x: x.split("_")[-2])
                df['source_set'] = 'ldc_amr3.0'

            df_all = df_all.merge(df[['id',f'{model}_amr']], on='id', how='left')

    df_all.to_csv(parsed_dir / "all_amrs.csv", index=False)
    print(df_all.shape)





if __name__ == "__main__":
    main()
    merge_gpt_amrs()