import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report
# def classify_content(prediction):
#     # 对“无毒”类别的识别扩展关键词
#     non_toxic_keywords = ["无害", "无毒", "非有毒", "非有害", "安全", "正面", "友好"]
#     toxic = 0 if any(keyword in prediction for keyword in non_toxic_keywords) else 1

#     # 对“有毒类型”的识别扩展关键词和范围
#     toxic_type = 0
#     if any(word in prediction for word in ["仇恨言论", "仇恨", "种族清洗"]):
#         toxic_type = 2
#     elif any(word in prediction for word in ["攻击性语言", "敌意", "辱骂", "一般攻击性语言"]):
#         toxic_type = 1

#     # 对“表达方式”的识别扩展关键词和范围
#     expression = 0
#     if any(word in prediction for word in ["显性仇恨", "公然", "直白"]):
#         expression = 1
#     elif any(word in prediction for word in ["隐性仇恨", "含蓄", "暗示"]):
#         expression = 2
#     elif any(word in prediction for word in ["报道", "非仇恨"]):
#         expression = 3

#     target = [0] * 5  # 确保target始终有5个元素
#     if "LGBTQ" in prediction or "同性恋" in prediction or "变性" in prediction:
#         target[0] = 1
#     if "地区" in prediction or "省份" in prediction or "城市" in prediction or "国家" in prediction:
#         target[1] = 1
#     if "性别主义" in prediction or "性别歧视" in prediction or "女性" in prediction or "男性" in prediction:
#         target[2] = 1
#     if "种族主义" in prediction or "种族" in prediction or "民族" in prediction or "族裔" in prediction:
#         target[3] = 1
#     if "其他" in prediction or "其他群体" in prediction or "不同群体" in prediction:
#         target[4] = 1

#     return toxic, toxic_type, expression, target
def classify_content(prediction):
    # 对“无毒”类别的识别扩展关键词
    non_toxic_keywords = ["无害", "无毒", "非有毒", "非有害", "安全", "正面", "友好", "健康", "积极"]
    toxic = 0 if any(keyword in prediction for keyword in non_toxic_keywords) else 1

    # 对“有毒类型”的识别扩展关键词和范围
    toxic_type = 0
    if any(word in prediction for word in ["仇恨言论", "种族清洗", "歧视性言论", "激进"]):
        toxic_type = 2
    elif any(word in prediction for word in ["攻击性语言", "敌意", "辱骂", "挑衅"]):
        toxic_type = 1

    # 对“表达方式”的识别扩展关键词和范围
    expression = 0
    if any(word in prediction for word in ["显性仇恨", "直言不讳", "公开侮辱"]):
        expression = 1
    elif any(word in prediction for word in ["隐性仇恨", "含蓄", "暗示", "微妙的偏见", "轻蔑"]):
        expression = 2
    elif "报道" in prediction :
        expression = 3

    target = [0] * 5
    if any(word in prediction for word in ["LGBTQ", "同性恋", "变性", "双性恋", "跨性别"]):
        target[0] = 1
    if any(word in prediction for word in ["地区", "省份", "城市", "国家", "东南亚", "中东", "非洲"]):
        target[1] = 1
    if any(word in prediction for word in ["性别主义", "性别歧视", "女性", "男性", "女权", "男权"]):
        target[2] = 1
    if any(word in prediction for word in ["种族主义", "种族", "民族", "族裔", "种族偏见"]):
        target[3] = 1
    if any(word in prediction for word in ["其他", "其他群体", "不同群体", "宗教", "身份"]):
        target[4] = 1

    return toxic, toxic_type, expression, target





def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def batch_predict(texts, system_prompt_content, tokenizer, llm, sampling_params):
    batch_input = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt_content},
             {"role": "user", "content": text}],
            tokenize=False, 
            add_generation_prompt=True
        ) for text in texts
    ]
    outputs = llm.generate(batch_input, sampling_params)
    predictions = [output.outputs[0].text for output in outputs]
    return predictions

def calculate_metrics(predictions, labels):
    from sklearn.metrics import classification_report
    report = classification_report(labels, predictions, output_dict=True)
    return report
 
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def main(model_path, json_file_path):
    data = load_data(json_file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.1, top_p=1.0, repetition_penalty=1.05, max_tokens=256)
    system_prompt_content = """
作为一名内容审核专家，您需要评估以下帖子内容，并对每个类别选择一个具体的描述：

内容评估：
1.判断此帖子是否是有害或无害。
2.有害类型：无害/普通攻击性语言/仇恨言论
3.表达方式：/非仇恨/显性仇恨言论/隐性仇恨言论/报道
4.目标群体（可以多选）：标记适用的描述：LGBTQ/地区/性别主义/种族主义/其他/非仇恨
请根据帖子的内容和语境，在每个类别中选择最合适的描述。这将帮助我们更准确地理解帖子的性质和潜在的影响。
    """ 

    texts = [item['content'] for item in data]
    predictions = batch_predict(texts, system_prompt_content, tokenizer, llm, sampling_params)

    if len(predictions) != len(texts):
        raise ValueError(f"预测结果数量 {len(predictions)} 与输入文本数量 {len(texts)} 不匹配")

    # 初始化列表用于数据保存
    results = []

    for idx, (prediction, item) in enumerate(zip(predictions, data)):
        pred_toxic, pred_toxic_type, pred_expression, pred_target = classify_content(prediction)
        
        results.append({
            'Content': item['content'],
            'Prediction': prediction,
            'Actual_Toxic': item['toxic'],
            'Predicted_Toxic': pred_toxic,
            'Actual_Toxic_Type': item['toxic_type'],
            'Predicted_Toxic_Type': pred_toxic_type,
            'Actual_Expression': item['expression'],
            'Predicted_Expression': pred_expression,
            'Actual_Target_LGBTQ': item['target'][0],
            'Predicted_Target_LGBTQ': pred_target[0],
            'Actual_Target_Region': item['target'][1],
            'Predicted_Target_Region': pred_target[1],
            'Actual_Target_Sexism': item['target'][2],
            'Predicted_Target_Sexism': pred_target[2],
            'Actual_Target_Racism': item['target'][3],
            'Predicted_Target_Racism': pred_target[3],
            'Actual_Target_Others': item['target'][4],
            'Predicted_Target_Others': pred_target[4]
        })

    df = pd.DataFrame(results)
    df.to_csv('prediction_results.csv', index=False)

    # 输出分类报告
    print("\nToxic Classification Report:")
    print(classification_report([item['toxic'] for item in data], [result['Predicted_Toxic'] for result in results]))
    
    print("\nToxic Type Classification Report:")
    print(classification_report([item['toxic_type'] for item in data], [result['Predicted_Toxic_Type'] for result in results]))
    
    print("\nExpression Classification Report:")
    print(classification_report([item['expression'] for item in data], [result['Predicted_Expression'] for result in results]))

    target_names = ['LGBTQ', 'Region', 'Sexism', 'Racism', 'Others']
    for name in target_names:
        print(f"\n{name} Classification Report:")
        current_labels = [item['target'][target_names.index(name)] for item in data]
        current_preds = [result['Predicted_Target_' + name] for result in results]
        print(classification_report(current_labels, current_preds, target_names=['Non-' + name, name]))
        print(f"{name} Accuracy: {accuracy_score(current_labels, current_preds):.4f}")

    # 总体目标分类的正确性
    overall_correct = [all(item['target'][i] == result['Predicted_Target_' + target_names[i]] for i in range(5)) for item, result in zip(data, results)]
    overall_accuracy = accuracy_score([True] * len(overall_correct), overall_correct)

    print("\nOverall Target Classification Accuracy:")
    print(f"Accuracy: {overall_accuracy:.4f}")
# 总体目标分类的正确性
    overall_labels = [any(item['target']) for item in data]
    overall_preds = [any([result['Predicted_Target_' + name] for name in target_names]) for result in results]
    print("\nOverall Target Classification Report:")
    print(classification_report(overall_labels, overall_preds, target_names=['Non-target', 'Target']))
    print(f"Overall Accuracy: {accuracy_score(overall_labels, overall_preds):.4f}")
    print("\nTotal Predictions Made: ", len(predictions))
    print("Total Texts Processed: ", len(texts))
    # 使用classification_report计算准确率、召回率和F1分数
    print("显性隐性")
    expression_labels = [item['expression'] for item in data]
    expression_preds = [result['Predicted_Expression'] for result in results]
    metrics = classification_report(expression_labels, expression_preds, labels=[1, 2], target_names=['显性仇恨言论', '隐性仇恨言论'], output_dict=True)
    
    print("Classification Report for Hate Speech Expression:")
    for label, scores in metrics.items():
        if isinstance(scores, dict):
            print(f"\nLabel: {label}")
            print(f"Precision: {scores['precision']:.4f}")
            print(f"Recall: {scores['recall']:.4f}")
            print(f"F1-Score: {scores['f1-score']:.4f}")

        
if __name__ == "__main__":
    model_path = "/hpc2hdd/home/jzhao815/LLaMA-Factory/saves/Qwen1.5-4B/freeze/newgen6"  # 替换为您的模型路径
    json_file_path = "toxicn.json"  # JSON文件路径
    main(model_path, json_file_path)
