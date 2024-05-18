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
)

def send_chat_message(prompt, user_text):
    try:
        completion = client.chat.completions.create(
            model="/hpc2hdd/home/jzhao815/model/shenzhi-wang/Llama3-8B-Chinese-Chat",
            max_tokens=1024,
            temperature=0.1,
            top_p=1.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API请求失败: {e}")
        return None

def generate_explanations_and_write_to_json(input_csv_path, output_json_path):
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as file:
            explanations = json.load(file)
        processed_texts = {ex['input'] for ex in explanations}
    else:
        explanations = []
        processed_texts = set()

    total_rows = 0  
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1  

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing", total=total_rows, unit="row"):
            text = row['text']
            if text in processed_texts:
                continue

            label = row['label']
            toxicity = '该评论被认定为\"无毒\"' if label == '0' else '该评论被认定为\"有毒\"'
            system_prompt = f"""作为一名专业的内容审核员，您的任务是判断以下帖子是否具有毒性。请遵循以下步骤进行评估，并提供详细的解释：
1.阅读并理解帖子内容：仔细阅读帖子，确保您理解了作者的意图和表达的情感。
2.检查身份特征提及（Q1）：在文本中寻找是否有提及任何个体或身份特征，例如种族、国籍、年龄、政治立场、宗教、残疾、地域、性别或职业。
3.识别不当语言（Q2）：判断文本中是否存在羞辱性、贬低性词语或脏话。
4.关联身份特征与不当语言（Q3）：如果文本中存在不当语言，分析这些词语或短语是否明确指向或针对在Q1中提到的特定身份或个体。
5.评估攻击性（Q4）：如果不当语言确实指向了特定身份或个体，进一步判断这些表达是否构成对该身份或个体的攻击。
6.综合判断：基于上述分析，综合考虑文本内容、语境、可能产生的影响，并判定该帖子是否具有毒性。
7.提供解释：给出您的判定原因，包括您在评估过程中观察到的关键点，以及这些点如何支持您的判断。提供关于以下评论的分析：{text}"""
            user_text = f"评论内容：{text}\n标签：{toxicity}"
            
            explanation = send_chat_message(system_prompt, user_text)
            if explanation is None:
                continue  

            explanations.append({
                "instruction": """作为一名专业的内容审核员，您的任务是判断以下帖子是否具有毒性。请遵循以下步骤进行评估，并提供详细的解释：
1.阅读并理解帖子内容：仔细阅读帖子，确保您理解了作者的意图和表达的情感。
2.检查身份特征提及（Q1）：在文本中寻找是否有提及任何个体或身份特征，例如种族、国籍、年龄、政治立场、宗教、残疾、地域、性别或职业。
3.识别不当语言（Q2）：判断文本中是否存在羞辱性、贬低性词语或脏话。
4.关联身份特征与不当语言（Q3）：如果文本中存在不当语言，分析这些词语或短语是否明确指向或针对在Q1中提到的特定身份或个体。
5.评估攻击性（Q4）：如果不当语言确实指向了特定身份或个体，进一步判断这些表达是否构成对该身份或个体的攻击。
6.综合判断：基于上述分析，综合考虑文本内容、语境、可能产生的影响，并判定该帖子是否具有毒性。
7.提供解释：给出您的判定原因，包括您在评估过程中观察到的关键点，以及这些点如何支持您的判断。""",
                "input": text,
                "output": explanation + toxicity
            })
            processed_texts.add(text)

            if len(explanations) % 100 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as file:
                    json.dump(explanations, file, ensure_ascii=False, indent=4)

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(explanations, file, ensure_ascii=False, indent=4)

input_csv_path = 'train.csv'  
output_json_path = 'CHSD.json'  
generate_explanations_and_write_to_json(input_csv_path, output_json_path)
