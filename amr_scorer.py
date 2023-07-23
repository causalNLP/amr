import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
import re
import argparse
from tqdm import tqdm
import warnings
import smatch
from efficiency.function import shell
import argparse


# from efficiency.log import fwrite, fread
current_dir = Path(__file__).parent.resolve()
data_dir = Path(__file__).parent.resolve() / "data"
parsed_dir  = data_dir / "parsed_amrs"
root_dir = Path(__file__).parent.parent.resolve()
output_dir = data_dir / "outputs"
amr_eval_pack = os.path.join(root_dir, "amr-evaluation")


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


def premise_generator(premise_list):
    for premise in premise_list:
        yield premise.strip()

def hypothesis_generator(hypothesis_list):
    for hypothesis in hypothesis_list:
        yield hypothesis.strip()


def compute_smatch_for_pairs(premise_list, hypothesis_list):
    scores = []
    invalid_amr = []
    invalid_type = []
    invalid_count = 0
    for premise, hypothesis in tqdm(zip(premise_list, hypothesis_list)):
        if premise is None or hypothesis is None or premise is np.nan or hypothesis is np.nan:
            scores.append(None)  # Set the score to None if premise or hypothesis is None
            continue
        try:
            with warnings.catch_warnings(record=True) as w:
                smatch.match_triple_dict.clear()
                best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(premise, hypothesis)
                precision, recall, score = smatch.compute_f(best_match_num, test_triple_num, gold_triple_num)
                score = min(score, 1)
                scores.append(score)
                smatch.match_triple_dict.clear()
                # Check if any warnings were captured
                if w:
                    # If so, append the warning message to invalid_type
                    # invalid_type.append(str(w[-1].message))
                    invalid_type.append(None)
                    invalid_amr.append(0)
                    invalid_count += 1
                else:
                    invalid_amr.append(0)
                    invalid_type.append(None)
        except Exception as e:
            invalid_amr.append(1)
            invalid_type.append(str(e))
            invalid_count += 1
            scores.append(-1)  # or some other default score

    valid_scores = [score for score in scores if score is not None and score != -1]
    mean_score = sum(valid_scores) / len(valid_scores)
    print(f'Number of AMR parsed', len(invalid_type))
    print('Number of invalid AMRs in valid part of hypothesis_list:', invalid_count)
    print('Number of valid AMRs in valid part of hypothesis_list:', len(valid_scores))
    print('Mean smatch score:', mean_score)
    return scores, invalid_count, invalid_amr, invalid_type


def instance_relation_match(amr1_input, amr2_input):
  if amr1_input is None or amr2_input is None:
    return 0,0

  try:
      amr1 = smatch.amr.AMR.parse_AMR_line(amr1_input)
      amr2 = smatch.amr.AMR.parse_AMR_line(amr2_input)
  except Exception as e:
      return 0,0
  if amr1 is None or amr2 is None:
    return 0,0
  instance_t, relation_t = amr1.get_triples2()
  instance_t2, relation_t2 = amr2.get_triples2()
  def quantify_similarity(list1, list2):
      matches = sum(1 for a, b in zip(list1, list2) if a == b)
      avg_length = (len(list1) + len(list2)) / 2
      return matches / avg_length

  instance_match = quantify_similarity(instance_t, instance_t2)
  relation_match = quantify_similarity(relation_t, relation_t2)

  return instance_match, relation_match






###### for paired amrs ######
def get_3_amr_features(df, amr_pred ='premise_amr', amr_gold='hypothesis_amr'):
  '''Given a df containing columns ['premise_amr','hypothesis_amr'],
  add three more columns to df ,['smatch_score','instance_match', 'relation_match']'''
  parser = amr_pred.replace("_amr", "")
  premise_list = df[amr_gold].tolist()
  hypothesis_list = df[amr_pred].tolist()

  df[f'{parser}_smatch'], invalid_counts, _, _= compute_smatch_for_pairs(premise_list, hypothesis_list)
  # mean_score = df[df[f'{parser}_smatch'] != -1].dropna().mean()

  return df






def scoring_parser_smatch(parser, save = True, file_path = parsed_dir / 'all_amrs.csv', test_only = False, subset = None):
    df = pd.read_csv(file_path)
    og_len = len(df)
    if test_only:
        df = df[df['split'] == 'test']
        save = False

    if subset is not None:
        save = False
        if subset in ['bio','bio_amr']:
            df = df[df['source_set'] == 'bio_amr0.8']
        elif subset in ['ldc','ldc_amr']:
            df = df[df['source_set'] == 'ldc_amr3.0']
        else:
            print('Subset not found, eval on all data')

    df[f'{parser}_amr'] = df[f'{parser}_amr'].apply(lambda x: re.sub('~\d+', '', x) if pd.notnull(x) else x)

    df = get_3_amr_features(df, amr_pred=f'{parser}_amr', amr_gold='gold_amr')
    if save and len(df) == og_len:
        df.to_csv(parsed_dir / f'{parser}_smatch.csv', index=False)

    print('Valid Smatch', parser,
          f"on {'all' if subset is None else subset} data {'test split' if test_only or 'gpt' in parser else 'all splits'}")
    valid_df = df[df[f'{parser}_smatch'] != -1].dropna()
    mean_score = valid_df[f'{parser}_smatch'].mean()
    print("Valid mean smatch: ", mean_score)
    return df


def comprehensive_eval(parser, save = True, file_path = parsed_dir / 'all_amrs.csv', test_only = False, subset = None):
    df = pd.read_csv(file_path)
    if test_only:
        df = df[df['split'] == 'test']

    if subset is not None:
        if subset in ['bio','bio_amr']:
            df = df[df['source_set'] == 'bio_amr0.8']
        elif subset in ['ldc','ldc_amr']:
            df = df[df['source_set'] == 'ldc_amr3.0']
        else:
            print('Subset not found, eval on all data')
    df = df[~df[f'{parser}_smatch'].isna()]
    df = df[~df[f'{parser}_smatch'].isin([-1])]
    df = df[~df[f'{parser}_smatch'].isnull()]


    df[f'{parser}_amr'] = df[f'{parser}_amr'].apply(lambda x: re.sub('~\d+', '', x) if pd.notnull(x) else x)
    df[f'{parser}_amr'] = df[f'{parser}_amr'].apply(parse_string_simple)
    print(f'Number of valid AMR', len(df))
    with open(f'{data_dir}/gold.txt', 'w') as f:
        for item in df['gold_amr']:
            f.write("%s\n\n" % item)

    with open(f'{data_dir}/{parser}_amr.txt', 'w') as f:
        for item in df[f'{parser}_amr']:
            f.write("%s\n\n" % item)

    cmd = f'cd {amr_eval_pack} && ./evaluation.sh {data_dir}/gold.txt {data_dir}/{parser}_amr.txt'
    stdout, stderr = shell(cmd)
    print('Comprehensive eval for', parser, f"on {'all' if subset is None else subset} data {'test split' if test_only or 'gpt' in parser else 'all splits'}")
    print(stdout)
    if save:
        with open(parsed_dir /f'{parser}_eval.txt', 'w') as f:
            f.write(stdout)
    return stdout, stderr





def scoring_gpt_smatch():
    df_ldc_test = pd.read_csv(f'{data_dir}/ldc_test.csv')
    read_in = output_dir / 'gpt4-0613_ldc_amr_2.0.csv'
    df = pd.read_csv(read_in)
    print(df.shape)
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
    # df.loc[:, 'gpt_amr'] = df['gpt-3.5-turbo-0613_amr'].apply(lambda x: x.replace("( (", "((").replace(") )", "))"))

    # find the true amr in ldc_test by id
    if 'true_amr' not in df.columns:
        df = df.merge(df_ldc_test[['id', 'true_amr']], on='id', how='left')
    # df = df[~df['gpt4_amr'].isna()]
    # df = df[~df['gpt-3.5-turbo-0613_amr'].isna()]
    df = df[~df['gpt4-0613_amr'].str.contains('None')]
    df.loc[:, 'gpt_amr'] = df['gpt4-0613_amr'].apply(parse_string_simple)
    # df.loc[:, 'gpt_amr'] = df['gpt_amr'].apply(lambda x: x.replace("unknown",":name (n / name :op1 'Unknown')"))
    # df.loc[:, 'gpt_amr'] = df['gpt_amr'].apply(extract_amr)
    # df.loc[:, 'gpt3_amr'] = df['gpt3_amr'].apply(balance_parentheses)
    df.loc[:, 'true_amr'] = df['true_amr'].apply(lambda x: re.sub("~e.\d+", "", x))
    df = get_3_amr_features(df, amr_pred='gpt_amr', amr_gold='true_amr')
    print(df['smatch_score'].mean())

    # df.to_csv(read_in, index=False)

    # to_inspect_path = data_dir/'gpt4-0613_invalid_amrs.csv'
    # to_inspect = df[df['invalid_amr'] == 1]
    # to_inspect.to_csv(to_inspect_path, index=False)


def get_amr_features(input_file):
    df = pd.read_csv(input_file)
    df.loc[:, 'amr'] = df['amr'].apply(extract_amr)
    df.loc[:, 'amr'] = df['amr'].apply(parse_string)
    df.loc[:, 'true_amr'] = df['true_amr'].apply(lambda x: re.sub("~e.\d+", "", x))
    df = get_3_amr_features(df, amr_pred='amr', amr_gold='true_amr')
    return df


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate smatch score for a parser')
    parser.add_argument('--parser', type=str, default='elit', help='parser name')
    parser.add_argument('--test_only', action = 'store_true', default=False, help='whether to evaluate on test set only')
    parser.add_argument('--subset', type=str, default=None, help='subset of data to evaluate on')
    args = parser.parse_args()
    return args



def main(parser, test_only = False, subset = None):
    print(f'evaluate {parser}')
    scoring_parser_smatch(parser, test_only=test_only, subset=subset)
    # comprehensive_eval(parser, test_only=test_only, subset=subset)

if __name__ == '__main__':
    args = get_args()
    parser = args.parser
    test_only = args.test_only
    parser = parser.replace(' ', '_').replace('-', '_')
    main(parser, test_only=args.test_only, subset=args.subset)