import csv
from openai import OpenAI
from tqdm import tqdm
import json
import os

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=200.0,
)

def send_chat_message(prompt, user_text):
    completion = client.chat.completions.create(
        model="/hpc2hdd/home/jzhao815/model/qwen/Qwen1.5-110B-Chat-GPTQ-Int4",
        max_tokens=1024,
        temperature=0.1,
        top_p=1.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ]
    )
    return completion.choices[0].message.content


def load_or_initialize_json(output_json_path):
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        return []
import logging
logging.basicConfig(filename='chsd_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def generate_explanations_and_write_to_json(input_csv_path, output_json_path):
    explanations = load_or_initialize_json(output_json_path)  
    existing_texts = {ex['input'] for ex in explanations}  

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        total_rows = sum(1 for row in reader)  

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing", total=total_rows, unit="row"):
            text = row['text']
            if text in existing_texts:  # skip
                continue
            label = row['label']
            toxicity = '该评论被认定为"无毒"' if label == '0' else '该评论被认定为"有毒"'
            system_prompt = f"""任务描述：您需要根据给定的评论和标签，生成描述评论的主题和表达方式的数据。

操作步骤：
1. 阅读评论内容：仔细阅读提供的评论。
2. 使用标签：基于给定的“{toxicity}”标签来分析评论。
3. 生成主题：确定并指出这条帖子主要关联的主题是什么（LGBTQ、性别歧视、种族歧视、地区）。
4. 评估表达方式：描述评论的表达方式，例如是否属于“显性仇恨言论”、“隐性仇恨言论”或“报道”等。
5. 评估毒性类别：‘非有害的’或‘一般攻击性语言’或‘仇恨言论’。
6. 构造输出：生成一个格式化的输出，详细说明评论的毒性、主题和表达方式。
如果你觉得标签是错的，请你再回答中包含"标签有误" 。
输出示例：该评论被认为是"有毒"，涉及的主题为‘LGBTQ’。表达方式被视为‘显性仇恨言论’，它属于‘仇恨言论’。"""


            user_text = f"评论内容：{text}\n标签：{toxicity}"
            explanation = send_chat_message(system_prompt, user_text)
            if "标签有误" in explanation:
                continue           
            if explanation:
                explanations.append({
                    "instruction": "作为一名内容审核专家，您需要评估下列帖子内容：\n1. 请判断这条帖子是否含有令人不适的内容或有害言论。\n2. 确定并指出这条帖子主要关联的主题是什么（如性别、种族、地区等）。\n3. 评估并描述这条帖子的表达方式（例如是否包含显性或隐性的仇恨言论）。\n\n请在您的判断中考虑帖子的整体语境和表达的细节，以便做出准确的评估。例如，如果帖子以负面方式描绘了某一特定群体，即使没有直接使用侮辱性语言，也应视为有害内容。",
                    "input": text,
                    "output": explanation
                })
                existing_texts.add(text)

            if len(explanations) % 500 == 0:
                part_path = f"{output_json_path}_part{len(explanations)//1000}.json"
                with open(part_path, 'w', encoding='utf-8') as file:
                    json.dump(explanations, file, ensure_ascii=False, indent=4)
                logging.info(f"processing {len(explanations)} ,save to {part_path}")



    if explanations:
        with open(output_json_path, 'w', encoding='utf-8') as file:
            json.dump(explanations, file, ensure_ascii=False, indent=4)
        logging.info(f"All data processing is completed")
input_csv_path = 'CHSDtrain.csv' 
output_json_path = 'CHSDnew.json' 
generate_explanations_and_write_to_json(input_csv_path, output_json_path)
