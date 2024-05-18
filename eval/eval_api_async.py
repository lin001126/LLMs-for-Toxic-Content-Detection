import argparse
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from datetime import datetime, timedelta
from openai import AsyncOpenAI
import asyncio

import aiofiles

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"
client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=20.0,
)
from datetime import datetime, timedelta
import json
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd

async def send_chat_message(system_prompt, user_text):
    print(f"Attempting to send request for text: {user_text[:30]}...")  
    try:
        completion = await client.chat.completions.create(
            model="/hpc2hdd/home/jzhao815/model/qwen4b_llamapro/checkpoint-700",
            max_tokens=1024,
            temperature=0.1,
            top_p=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        print(f"Completion received for text: {user_text[:30]}...")  
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call for text: {user_text[:30]}... Error: {str(e)}")  
        return None

from asyncio import Semaphore

async def process_file(df, system_prompt, output_file_path, sem):
    print(f"Processing file: {output_file_path}")
    tasks = []
    for text in df['post'].tolist():
        async with sem:  
            task = send_chat_message(system_prompt, text)
            result = await task  
            tasks.append(result)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for text, prediction in zip(df['post'].tolist(), tasks):
            file.write(json.dumps({'text': text, 'predict': prediction}, ensure_ascii=False) + '\n')
    print(f"Finished processing file: {output_file_path}")

async def main(data_dir):
    system_prompt_content = """
    作为一名内容审核专家，您需要评估下列帖子内容：
请判断这条帖子是否含有令人不适的内容或有害言论。
确定并指出这条帖子主要关联的主题是什么（如性别、种族、地区等）。
评估并描述这条帖子的表达方式（例如是否包含显性或隐性的仇恨言论）。
请确定并标记这条帖子所涉及的目标群体是哪些（如LGBTQ、地区、性别主义、种族主义、其他）。
请在您的判断中考虑帖子的整体语境和表达的细节，以便做出准确的评估。请你一步一步思考并生成思考过程，最后如果有毒请你输出“有毒”，否则输出“无毒”,下面是文本：
"""
    print("Starting processing...")
    start_time = datetime.now()

    tasks = []
    for filename in [f for f in os.listdir(data_dir) if f.endswith('.csv')]:
        print(f"Processing file: {filename}")
        csv_file_path = os.path.join(data_dir, filename)
        output_file_path = os.path.join(data_dir, f"{filename}_predictions.json")
        df = pd.read_csv(csv_file_path)
        task = process_file(df, system_prompt_content, output_file_path)
        tasks.append(task)

    await asyncio.gather(*tasks)

    end_time = datetime.now()
    total_time_seconds = (end_time - start_time).total_seconds()
    print(f"Total time taken: {str(timedelta(seconds=total_time_seconds))}")
    print("All tasks completed.")

datapath = "/hpc2hdd/home/jzhao815/eval/zh"
asyncio.run(main(datapath))







