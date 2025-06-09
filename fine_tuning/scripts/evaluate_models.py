#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型测试和对比脚本
测试微调后的QWEN和LLAMA模型在经济学问答任务上的表现
"""

import os
import torch
import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from rouge import Rouge
import jieba

def load_test_data(file_path, format_type="json"):
    """加载测试数据"""
    if format_type == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif format_type == "jsonl":
        data = []
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
    return data

def load_qwen_model(base_model_path, checkpoint_path):
    """加载微调后的QWEN模型"""
    print("加载QWEN模型...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model, tokenizer

def load_llama_model(base_model_path, checkpoint_path):
    """加载微调后的LLAMA模型"""
    print("加载LLAMA模型...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model, tokenizer

def generate_qwen_response(model, tokenizer, question, max_length=256):
    """使用QWEN模型生成回答"""
    prompt = f"<|im_start|>system\n你是一位专业的经济学助手，能够准确回答经济学相关问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码并提取回答部分
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("<|im_start|>assistant\n")[-1]
    
    return response.strip()

def generate_llama_response(model, tokenizer, question, max_length=256):
    """使用LLAMA模型生成回答"""
    prompt = f"### Instruction:\n请回答以下经济学问题：\n\n### Input:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码并提取回答部分
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:\n")[-1]
    
    return response.strip()

def calculate_rouge(pred, ref):
    """计算ROUGE分数"""
    rouge = Rouge()
    # 使用jieba分词
    pred_tokens = ' '.join(jieba.cut(pred))
    ref_tokens = ' '.join(jieba.cut(ref))
    
    try:
        scores = rouge.get_scores(pred_tokens, ref_tokens)[0]
        return scores
    except:
        return {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}

def evaluate_model(model, tokenizer, test_data, model_name, generate_func):
    """评估模型"""
    print(f"开始评估{model_name}模型...")
    
    total_time = 0
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    predictions = []
    
    for i, item in enumerate(test_data[:20]):  # 测试前20个样本
        if model_name == "QWEN":
            question = item.get("text", "").split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            reference = item.get("text", "").split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
        else:  # LLAMA
            question = item["input"] 
            reference = item["output"]
        
        start_time = time.time()
        prediction = generate_func(model, tokenizer, question)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        
        # 计算ROUGE分数
        rouge_score = calculate_rouge(prediction, reference)
        rouge_scores["rouge-1"].append(rouge_score["rouge-1"]["f"])
        rouge_scores["rouge-2"].append(rouge_score["rouge-2"]["f"])
        rouge_scores["rouge-l"].append(rouge_score["rouge-l"]["f"])
        
        predictions.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "rouge_scores": rouge_score
        })
        
        print(f"样本 {i+1}/20 完成")
    
    # 计算平均分数
    avg_rouge = {
        "rouge-1": sum(rouge_scores["rouge-1"]) / len(rouge_scores["rouge-1"]),
        "rouge-2": sum(rouge_scores["rouge-2"]) / len(rouge_scores["rouge-2"]),
        "rouge-l": sum(rouge_scores["rouge-l"]) / len(rouge_scores["rouge-l"])
    }
    
    avg_time = total_time / len(test_data[:20])
    
    return avg_rouge, avg_time, predictions

def main():
    # 配置
    qwen_base = "Qwen/Qwen2.5-3B-Instruct"
    llama_base = "meta-llama/Llama-3.2-3B-Instruct"
    
    qwen_checkpoint = "./fine_tuning/qwen/checkpoints"
    llama_checkpoint = "./fine_tuning/llama/checkpoints"
    
    qwen_test_data_path = "./fine_tuning/data/qwen/test.jsonl"
    llama_test_data_path = "./fine_tuning/data/alpaca/test.json"
    
    # 检查模型是否存在
    if not os.path.exists(qwen_checkpoint):
        print(f"QWEN检查点不存在: {qwen_checkpoint}")
        return
    
    if not os.path.exists(llama_checkpoint):
        print(f"LLAMA检查点不存在: {llama_checkpoint}")
        return
    
    # 加载测试数据
    qwen_test_data = load_test_data(qwen_test_data_path, "jsonl")
    llama_test_data = load_test_data(llama_test_data_path, "json")
    
    print(f"QWEN测试数据: {len(qwen_test_data)} 个样本")
    print(f"LLAMA测试数据: {len(llama_test_data)} 个样本")
    
    # 评估QWEN模型
    qwen_model, qwen_tokenizer = load_qwen_model(qwen_base, qwen_checkpoint)
    qwen_rouge, qwen_time, qwen_predictions = evaluate_model(
        qwen_model, qwen_tokenizer, qwen_test_data, "QWEN", generate_qwen_response
    )
    
    # 评估LLAMA模型
    llama_model, llama_tokenizer = load_llama_model(llama_base, llama_checkpoint)
    llama_rouge, llama_time, llama_predictions = evaluate_model(
        llama_model, llama_tokenizer, llama_test_data, "LLAMA", generate_llama_response
    )
    
    # 输出结果
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    
    print(f"\nQWEN模型:")
    print(f"  ROUGE-1: {qwen_rouge['rouge-1']:.4f}")
    print(f"  ROUGE-2: {qwen_rouge['rouge-2']:.4f}")
    print(f"  ROUGE-L: {qwen_rouge['rouge-l']:.4f}")
    print(f"  平均推理时间: {qwen_time:.2f}秒")
    
    print(f"\nLLAMA模型:")
    print(f"  ROUGE-1: {llama_rouge['rouge-1']:.4f}")
    print(f"  ROUGE-2: {llama_rouge['rouge-2']:.4f}")
    print(f"  ROUGE-L: {llama_rouge['rouge-l']:.4f}")
    print(f"  平均推理时间: {llama_time:.2f}秒")
    
    # 保存详细结果
    results = {
        "qwen": {
            "rouge_scores": qwen_rouge,
            "avg_time": qwen_time,
            "predictions": qwen_predictions
        },
        "llama": {
            "rouge_scores": llama_rouge,
            "avg_time": llama_time,
            "predictions": llama_predictions
        }
    }
    
    with open("./fine_tuning/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: ./fine_tuning/evaluation_results.json")
    
    # 展示一些样例
    print("\n" + "="*50)
    print("样例对比")
    print("="*50)
    
    for i in range(min(3, len(qwen_predictions))):
        print(f"\n样例 {i+1}:")
        print(f"问题: {qwen_predictions[i]['question']}")
        print(f"参考答案: {qwen_predictions[i]['reference']}")
        print(f"QWEN回答: {qwen_predictions[i]['prediction']}")
        print(f"LLAMA回答: {llama_predictions[i]['prediction']}")
        print("-" * 30)

if __name__ == "__main__":
    main() 