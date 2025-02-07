import pandas as pd
import os
import openai
from datasets import load_dataset
import subprocess
import shlex
import json
import re
import zipfile
import numpy as np
from tqdm import tqdm
from databench_eval import Runner, Evaluator, utils
openai.api_key = os.getenv("OPENAI_API_KEY")  

def load_sample_csv(dataset_id: str):
    try:
        
        df_sample = pd.read_parquet(f"data/competition/competition/{dataset_id}/sample.parquet")
        # df_sample = pd.read_parquet(f"data/competition/competition/{dataset_id}/all.parquet")
        return df_sample
    except Exception as e:
        return f"Error loading dataset {dataset_id}: {e}"


model_id = "gpt-4o-2024-11-20" # " gpt-4-turbo gpt-4o-2024-11-20"



def call_chatgpt_model(prompts):
    res_dict = {}
    res_list = []
    try:
        cur_prompt = {
            "model": model_id, 
            "messages": [{"role": "system", 
                        "content": "You are an expert in coding"},
                        {"role": "user", "content": prompts}],
            "n":3,
            "temperature": 0.7,
        }
        response = openai.ChatCompletion.create(**cur_prompt)
        res_list.extend(res['message']['content'] for res in response['choices'])
    except Exception as e:
        res_list.append(f"__CODE_GEN_ERROR__: {e}")
    res_dict= res_list
    return res_dict


def example_generator(question, dataset) -> str:

    df = load_sample_csv(dataset)
    return f'''
    You are a python code generation expert. Your goal is to complete the function provided to answer the question.
    Notes: 
- You only have access to pandas and numpy, and python.
- Pay attention to the type formatting.
- You cannot read files from disk.
- **Only output the completed python code with well format, I will fee you 100 dollars.**.
import pandas as pd
import numpy as np

def answer(df: pd.DataFrame):
    """Returns the answer to the question: {question} """
    df.columns = {list(df.columns)}
    df.dtypes = {df.dtypes}
    one record = {list(df.iloc[0])} 
    # Note that You must convert the categorical columns that are necessary for answering the questions, not all categorical columns in the df. Ensure that these columns are transformed into a computable numeric type or a comparable type based on the given record."
    ...'''

pattern = r'```python(.*?)```'


def example_postprocess(responses, dataset: str, loader):
    ans_list= []
    for res_code11 in responses:
        match = re.search(pattern, res_code11,  re.DOTALL | re.IGNORECASE)
        if match:
            res_code11 = match.group(1)
        else:
            pass
        try:
            df = loader(dataset)
            global ans

            exec_string = (
                res_code11.replace("```python", "").replace("```", "")
                + "\nans = answer(df)"
            )     
            local_vars = {"df": df, "pd": pd, "np": np}
            exec(exec_string, local_vars)

            ans = local_vars['ans']
            if isinstance(ans, np.bool_):
                ans = bool(ans)
            if isinstance(ans, pd.Series):
                ans = ans.tolist()
            elif isinstance(ans, pd.DataFrame):
                ans = ans.iloc[:, 0].tolist()
            ans_list.append(ans.split('\n')[0] if '\n' in str(ans) else ans)
        except Exception as e:
            print(f"__CODE_ERROR__: {e}")
            ans_list.append(None)
            # return f"__CODE_ERROR__: {e}"
    return ans_list
        

def main():

    qa_data = load_dataset('csv', data_files='data/competition/competition/test_qa.csv', split='train')
    print(len(qa_data))
    dataset = qa_data["dataset"]
    question = qa_data["question"]
    res_list = []
    for idx, (each_q, each_table) in tqdm(enumerate(zip(question, dataset)), total=len(question)):
        tmp_dict = {}
        prompt = example_generator(each_q, each_table)
        res_codes = call_chatgpt_model(prompt)
        final_answer = example_postprocess(res_codes, each_table, load_sample_csv)
        print(final_answer[0])
        tmp_dict['idx'] = idx
        tmp_dict['codes'] = res_codes
        tmp_dict['answer'] = final_answer
        res_list.append(tmp_dict)

    out_path = './prediction_codes_lites.json'
    with open(out_path, 'w') as fw:
        json.dump(res_list, fw, indent = 2)
if __name__ == "__main__":
    main()
