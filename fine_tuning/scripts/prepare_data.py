#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据准备脚本
将经济学问答对数据转换为微调格式
"""

import json
import pandas as pd
import jsonlines
from sklearn.model_selection import train_test_split
import os

def load_qa_data(file_path):
    """加载问答对数据"""
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

def convert_to_chat_format(data):
    """转换为对话格式，适用于指令微调"""
    formatted_data = []
    
    for item in data:
        # 构建对话格式
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的经济学助手，能够准确回答经济学相关问题。你的回答应该基于《经济学原理》教材内容，准确、清晰、有教育价值。"
                },
                {
                    "role": "user", 
                    "content": item["question"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ],
            "chapter": item.get("chapter", ""),
            "chapter_num": item.get("chapter_num", ""),
            "chunk_id": item.get("chunk_id", "")
        }
        formatted_data.append(conversation)
    
    return formatted_data

def create_alpaca_format(data):
    """创建Alpaca格式的数据，适用于LLAMA微调"""
    alpaca_data = []
    
    for item in data:
        alpaca_item = {
            "instruction": "请回答以下经济学问题：",
            "input": item["question"],
            "output": item["answer"],
            "chapter": item.get("chapter", ""),
            "chapter_num": item.get("chapter_num", "")
        }
        alpaca_data.append(alpaca_item)
    
    return alpaca_data

def create_qwen_format(data):
    """创建QWEN格式的数据"""
    qwen_data = []
    
    for item in data:
        # QWEN使用类似ChatML的格式
        text = f"<|im_start|>system\n你是一位专业的经济学助手，能够准确回答经济学相关问题。<|im_end|>\n"
        text += f"<|im_start|>user\n{item['question']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['answer']}<|im_end|>"
        
        qwen_item = {
            "text": text,
            "chapter": item.get("chapter", ""),
            "chapter_num": item.get("chapter_num", "")
        }
        qwen_data.append(qwen_item)
    
    return qwen_data

def split_data(data, test_size=0.2, val_size=0.1):
    """分割数据为训练、验证和测试集"""
    # 首先分出测试集
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    
    # 然后从训练集中分出验证集
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    return train, val, test

def save_data(data, output_path, format_type="json"):
    """保存数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format_type == "jsonl":
        with jsonlines.open(output_path, 'w') as writer:
            for item in data:
                writer.write(item)

def main():
    # 输入文件路径
    input_file = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
    
    # 输出目录
    output_dir = "fine_tuning/data"
    
    print("加载原始数据...")
    raw_data = load_qa_data(input_file)
    print(f"总共加载了 {len(raw_data)} 个问答对")
    
    # 分割数据
    print("分割数据为训练、验证和测试集...")
    train_data, val_data, test_data = split_data(raw_data)
    print(f"训练集: {len(train_data)} 个")
    print(f"验证集: {len(val_data)} 个")
    print(f"测试集: {len(test_data)} 个")
    
    # 创建不同格式的数据
    print("转换为不同格式...")
    
    # 1. 对话格式（通用）
    train_chat = convert_to_chat_format(train_data)
    val_chat = convert_to_chat_format(val_data)
    test_chat = convert_to_chat_format(test_data)
    
    # 2. Alpaca格式（LLAMA）
    train_alpaca = create_alpaca_format(train_data)
    val_alpaca = create_alpaca_format(val_data)
    test_alpaca = create_alpaca_format(test_data)
    
    # 3. QWEN格式
    train_qwen = create_qwen_format(train_data)
    val_qwen = create_qwen_format(val_data)
    test_qwen = create_qwen_format(test_data)
    
    # 保存数据
    print("保存数据...")
    
    # 通用对话格式
    save_data(train_chat, f"{output_dir}/chat/train.json")
    save_data(val_chat, f"{output_dir}/chat/val.json")
    save_data(test_chat, f"{output_dir}/chat/test.json")
    
    # Alpaca格式（LLAMA）
    save_data(train_alpaca, f"{output_dir}/alpaca/train.json")
    save_data(val_alpaca, f"{output_dir}/alpaca/val.json")
    save_data(test_alpaca, f"{output_dir}/alpaca/test.json")
    
    # QWEN格式
    save_data(train_qwen, f"{output_dir}/qwen/train.jsonl", "jsonl")
    save_data(val_qwen, f"{output_dir}/qwen/val.jsonl", "jsonl")
    save_data(test_qwen, f"{output_dir}/qwen/test.jsonl", "jsonl")
    
    print("数据准备完成！")
    print(f"数据保存在: {output_dir}")
    
    # 打印一些样本
    print("\n=== 样本数据 ===")
    print("对话格式样本:")
    print(json.dumps(train_chat[0], ensure_ascii=False, indent=2))
    
    print("\nAlpaca格式样本:")
    print(json.dumps(train_alpaca[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 