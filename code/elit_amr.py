import argparse
import csv
import os
import pandas as pd
import elit
from tqdm import tqdm

os.environ["HF_HOME"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"


def main(input_file, output_file):
    amr_parser = elit.load(elit.pretrained.amr.AMR3_BART_LARGE_EN)
    print('parser loaded', flush=True)
    df=pd.read_csv(input_file)
    for i in tqdm(range(0,df.shape[0],1), total = df.shape[0]):
        data=list(df['text_detok'].values)[i]
        idx=list(df['id'].values)[i]
        try:
            amr = amr_parser(data)
            with open(output_file, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerow([idx,data,amr])
            if i%100==0:
                print(f'Finished {i} sentences', flush = True)
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gets AMR from text')
    parser.add_argument('--input_file', type=str, default="/home/yuenchen/amr_computation/data/all_amrs.csv", help='the input csv file')
    parser.add_argument('--output_file', type=str, default='/home/yuenchen/amr_computation/data/elit_all_amrs.csv',  help='the output csv file')
    # parser.add_argument('--model', type=str, default='elit', help='the model name')
    args = parser.parse_args()
    main(args.input_file, args.output_file)

