import pandas as pd
from sqlalchemy import create_engine
import openai
import json
from tqdm import tqdm
import os
import pandas as pd
import openai
from datasets import load_dataset
import subprocess
import shlex
import zipfile
import numpy as np
from databench_eval import Runner, Evaluator, utils
openai.api_key = os.getenv("OPENAI_API_KEY")  
model_id = "gpt-4-turbo" # "gpt-4o-2024-11-20"
def load_sample_csv(dataset_id: str):
    try:
        df_sample = pd.read_parquet(f"data/competition/competition/{dataset_id}/sample.parquet")
        # df_sample = pd.read_parquet(f"data/competition/competition/{dataset_id}/all.parquet")
        return df_sample
    except Exception as e:
        return f"Error loading dataset {dataset_id}: {e}"

def create_database(data_df, table_name, engine):
    new_name = '_'.join(table_name.split('_')[1:])
    data_df.to_sql(new_name, engine, if_exists='replace', index=False)


def text_to_sql_conversion(user_query, table_name, table_scheme):
    system_prompt = f"""You are an expert SQL developer.  Convert the following English question to SQL to answer the question"""
    user_prompt = f"""current table name: {table_name}\n
    current table scheme: {table_scheme}\n
    current question: {user_query}\n
    Rules:
    1. Use SQLite syntax.
    2. Always use explicit SQL syntax (e.g., avoid shorthand like `=` for `IS`).
    3. Format the SQL clearly and only return the SQL query, no extra text.
    4. The generated SQL is used to answer the question.
    """
    
    cur_prompt = {
        "model": model_id, 
        "messages": [{"role": "system", 
                    "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
        "n":3,
        "temperature": 0.5,
    }
    response = openai.ChatCompletion.create(**cur_prompt)

    res_list = []
    for res in response.choices:
        res_list.append(res.message['content'].strip())
    return res_list

def execute_sql(engine, sql_querys):
    res_list = []
    with engine.connect() as conn:
        try:
            for sql_query in sql_querys:
                result = pd.read_sql_query(sql_query, conn)
                res_list.append(result)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    return res_list

def generate_natural_answer(question, results):
    ans_list = []
    for res in results:
        if isinstance(res, pd.DataFrame) and not res.empty:
            answer = f"The query result for '{question}' is:\n{res.to_string(index=False)}"
        else:
            answer = f"Could not retrieve results for the query: {question}"
        ans_list.append(answer)
    return ans_list


def main():
    qa_data = load_dataset('csv', data_files='data/competition/competition/test_qa.csv', split='train')
    engine = create_engine('sqlite:///sales.db', echo=False)
    print(len(qa_data))
    dataset = qa_data["dataset"]
    question = qa_data["question"]
    clean_tables = set(dataset)
    for each_table in clean_tables:
        df = load_sample_csv(each_table)
        create_database(df, each_table, engine)
    res_list = []
    for idx, (each_q, each_table) in tqdm(enumerate(zip(question, dataset)), total=len(question)):
        tmp_dict = {}
        df = load_sample_csv(each_table)
        table_scheme = ''
        for col, dtype in df.dtypes.items():
            table_scheme += f"{col}: {dtype}\n"
        try:
            new_name = '_'.join(each_table.split('_')[1:])
            generated_sqls = text_to_sql_conversion(each_q, new_name, table_scheme)
            print(f"Generated SQL:\n{generated_sqls}\n")
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            return
        
        gen_sqls = [gen_sql.replace('```', '').replace('sql', '') for gen_sql in generated_sqls]
        query_results = execute_sql(engine, gen_sqls)
        
        final_answer = generate_natural_answer(each_q, query_results)
        print(final_answer[0])
        tmp_dict['idx'] = idx
        tmp_dict['sql'] = generated_sqls
        tmp_dict['answer'] = final_answer
        res_list.append(tmp_dict)
    out_path = './prediction_sql_lites.json'
    with open(out_path, 'w') as fw:
        json.dump(res_list, fw, indent = 2)
if __name__ == "__main__":
    main()