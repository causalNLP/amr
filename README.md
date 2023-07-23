# Revisiting AMR Parsing in the Era of LLMs
This repo is for our paper "Revisiting Text-to-AMR Parsing in the Era of LLMs" (2023).

## File Structure

- `code/`:
  - Generate AMR by a BART-based parser: `amr_.py`
  - Generate AMR by OpenAI's API for GPT models: `gpt_amr.py`
  - Generate AMR by the elit package: `python elit_amr.py`
  - Merge all the generated AMRs by various parsers: `merge_parsed.py`
  - Evaluate the generated AMRs by SMATCH: `amr_scorer.py`
- `data/`:
  - Please download the data files from the [Google Drive folder](https://drive.google.com/drive/folders/1QxyJKi_OPM0HBFD59WaUxaVnmPA21jS_?usp=drive_link) (~200M)
  - All the generated AMRs: `parsed_amrs/all_amrs.csv`. The headers are `id,text,AMR3_structbart_L_amr,split,source_set,gold_amr,text_detok,elit_amr,elit_smatch,AMR3_structbart_L_smatch,gpt_3.5_turbo_0613_amr,gpt4_0613_amr,gpt_3.5_turbo_0613_smatch,gpt4_0613_smatch`

## Step 1. Environment Setup

Please install the environment by the following command:

```bash
pip install -r requirements.txt
```



## Step 2. Text-to-AMR Generation by Various Parsers

### Parser 1: Structured-BART Model
**Setup:** We use the BART-based AMR parser proposed by [Drozdov et al. (2022, NAACL)](https://arxiv.org/abs/2205.01464), and use the [`transition-amr-parser`](https://github.com/IBM/transition-amr-parser) codes. 

**Example Usage:** 

```bash
python code/amr_.py --input_file ./data/ldc_gold_amrs_clean.csv --output_file ./data/parsed_amrs/AMR3-structbart-L_ldc.
csv --model AMR3-structbart-L
```

### Parser 2: GPT-3.5-turbo-0613/GPT-4-0613

**Setup:** For reproducibility, we use the checkpoints of GPT3.5 and GPT4 on June 13, 2023 (namely `gpt-3.5-turbo-0613` and `gpt-4-0613`), and set the text generation temperature to 0. 

**Example Usage:** 

```bash
export OPENAI_API_KEY="$YourOpenaiKey" # Copy and paste your openai key here.
python code/gpt_amr.py --input_file ./data/ldc_gold_amrs_clean.csv --output_file ./data/parsed_amrs/elit_ldc.csv 
--model_version gpt-4-0613
```

### Parser 3: ELIT Package

**Setup:** We use the Emory Language and Information Toolkit [(`elit` package](https://github.com/emorynlp/elit)), and follow its very user-friendly way to [call the AMR parser](https://github.com/emorynlp/elit/blob/main/docs/abstract_meaning_parsing.md).

**Example Usage:** 

```bash
python code/elit_amr.py --input_file ./data/ldc_gold_amrs_clean.csv --output_file ./data/parsed_amrs/elit_ldc.csv
```




## Step 3. AMR Scorer
### Output File Preparation:

Merge all the generated AMRs by various parsers: `python merge_parsed.py`

Then, you will get the file `data/parsed_amrs/all_amrs.csv` (which can also be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/1QxyJKi_OPM0HBFD59WaUxaVnmPA21jS_?usp=drive_link)).

### Scorer 1) Get the overall SMATCH score

We use the `smatch` pip package, and calculate the overall SMATCH score as follows:

```bash
python code/amr_scorer.py --input_file ./data/all_amrs.csv --parser elit --smatch_only
```

Note 1: The original SMATCH calculation has an ongoing issue of outputting F1_score greater than 1 (also reported in its github issues). Therefore, we set SMATCH score for each AMR pair to be ```min(original SMATCH, 1)```. 

Note 2: For invalid AMR predictions such as those with a missing parentheses or duplicate node names (e.g., in the case of GPT-produced AMRs), we set the SMATCH score to -1. In our paper report, we report the percentage of valid AMRs, and then report the score over valid examples.

### Scorer 2) Get the fine-grained score report

**Setup:** We compute a set of fine-grained AMR metrics in addition to the traditional Smatch code, following the evaluation framework of [Damonte et al. (2017, EACL)](https://arxiv.org/pdf/1608.06111.pdf) in the [`amr-evaluation` repo](https://github.com/mdtux89/amr-evaluation).

**Example Usage:** 

```bash
python code/amr_scorer.py --input_file ./data/all_amrs.csv --parser elit
```

You can change `--parser elit` to other parser names to get the scores for other parsers.