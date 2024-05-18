import argparse
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from datetime import datetime, timedelta

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def label_to_numeric(label):
    """Convert labels to numeric: non-toxic content is 0, toxic content is 1."""
    # Ensure label is a string and preprocess it
    label = str(label).lower().strip()  # Convert to lower case and strip whitespaces
    
    # List of non-toxic indicators, including both English and Chinese
    non_toxic_indicators = ["non-toxic", "not toxic", "非-toxic", "无害", "非有害","无毒","非有毒",'不仇恨',"安全", "正面", "友好", "健康", "积极"]
    
    # Check if any non-toxic indicator is present in the label
    if any(indicator in label for indicator in non_toxic_indicators):
        return 0
    else:
        return 1



def calculate_metrics(file_path):
    TP, FP, TN, FN = 0, 0, 0, 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            label_numeric = data['label']
            predict_numeric = label_to_numeric(data['predict'])
            
            if label_numeric == 1:
                if predict_numeric == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if predict_numeric == 1:
                    FP += 1
                else:
                    TN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

from datetime import datetime, timedelta
import os
import json
import pandas as pd
from tqdm import tqdm
import time  
def send_chat_message(system_prompt, user_text):
    completion = client.chat.completions.create(
        model="/hpc2hdd/home/jzhao815/model/qwen4b_llamapro/checkpoint-700",
        max_tokens=1024,temperature=0.1, top_p=1.0,  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )
    # print(completion.choices[0].message)

    return completion.choices[0].message.content




def main(model_path, data_dir):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = os.path.basename(model_path)
    output_dir = os.path.join("json", f"{now}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    results_summary = ""
    system_prompt_content = """
    作为一名内容审核专家，您需要评估下列帖子内容：
请判断这条帖子是否含有令人不适的内容或有害言论。
确定并指出这条帖子主要关联的主题是什么（如性别、种族、地区等）。
评估并描述这条帖子的表达方式（例如是否包含显性或隐性的仇恨言论）。
请确定并标记这条帖子所涉及的目标群体是哪些（如LGBTQ、地区、性别主义、种族主义、其他）。
请在您的判断中考虑帖子的整体语境和表达的细节，以便做出准确的评估。请你一步一步思考并生成思考过程，最后如果有毒请你输出“有毒”，否则输出“无毒”,下面是文本：
"""

    start_time = datetime.now()

    total_samples_processed = 0
    for filename in tqdm([f for f in os.listdir(data_dir) if f.endswith('.csv')]):
        print(f"Currently processing dataset: {filename}")
        csv_file_path = os.path.join(data_dir, filename)
        output_file_path = os.path.join(output_dir, filename.replace(".csv", "_predictions.json"))

        df = pd.read_csv(csv_file_path)
        texts = df['post'].tolist()
        labels = df['label'].tolist()
        predictions = [send_chat_message(system_prompt_content, text) for text in texts]

        with open(output_file_path, 'w', encoding='utf-8') as file:  
            for text, prediction, label in zip(texts, predictions, labels):
                file.write(json.dumps({'text': text, 'predict': prediction, 'label': label}, ensure_ascii=False) + '\n')
        accuracy, precision, recall, f1 = calculate_metrics(output_file_path)
        results_summary += f"{filename} - Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%\n"

        total_samples_processed += len(texts)

    end_time = datetime.now()
    total_time_seconds = (end_time - start_time).total_seconds()
    samples_per_second = total_samples_processed / total_time_seconds if total_time_seconds > 0 else 0
    results_summary += f"\nTotal samples processed: {total_samples_processed}\n"
    results_summary += f"Total time taken: {str(timedelta(seconds=total_time_seconds))}\n"

    results_summary += f"Samples per second: {samples_per_second:.2f}\n"

    results_file_path = os.path.join(output_dir, "all_datasets_results_summary.txt")
    with open(results_file_path, 'w', encoding='utf-8') as results_file:
        results_file.write(results_summary)

    print(f"All metrics have been saved to: {results_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of a particular model on multiple datasets and compute performance metrics")
    parser.add_argument("-modelpath", type=str, required=True, help="modelpath")
    args = parser.parse_args()

    datapath = "/hpc2hdd/home/jzhao815/eval/zh"

    main(args.modelpath, datapath)






