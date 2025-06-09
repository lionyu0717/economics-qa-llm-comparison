#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成1000个经济学问答对的脚本
修复JSON解析问题并确保达到目标数量
"""

import os
import sys
import json
import random
import time
import re
from tqdm import tqdm
import jsonlines
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API 相关配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "")

def create_qa_prompt_messages(text, num_questions=50):
    """创建用于生成问答对的提示消息列表"""
    system_prompt = """你是一位专业的经济学教授，正在为《经济学原理》教材创建高质量的问答对。

请根据提供的文本内容，创建指定数量的问答对。要求：

1. 问题要涵盖不同认知层次：定义、解释、应用、分析、比较等
2. 答案必须准确且完全基于文本内容
3. 问题要具体明确，避免过于宽泛
4. 确保问答对具有教育价值

请严格按照以下JSON格式输出，不要包含任何其他内容：
[
  {
    "question": "问题文本",
    "answer": "答案文本"
  }
]"""

    user_prompt = f"""文本内容：
{text}

请生成 {num_questions} 个高质量的问答对，严格按照JSON数组格式输出。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def clean_json_response(content):
    """清理并提取JSON内容"""
    # 移除markdown代码块标记
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    
    # 查找JSON数组
    json_match = re.search(r'\[[\s\S]*\]', content, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return content.strip()

def call_api_with_retry(messages, model="google/gemini-2.0-flash-001", temperature=0.3, max_retries=3):
    """调用API并重试"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    headers = {}
    if YOUR_SITE_URL:
        headers["HTTP-Referer"] = YOUR_SITE_URL
    if YOUR_SITE_NAME:
        headers["X-Title"] = YOUR_SITE_NAME
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4000,
                extra_headers=headers if headers else None
            )
            
            content = completion.choices[0].message.content.strip()
            
            # 清理和解析JSON
            json_content = clean_json_response(content)
            qa_pairs = json.loads(json_content)
            
            if isinstance(qa_pairs, list):
                return qa_pairs
            else:
                print(f"警告：返回的不是列表格式")
                return []
                
        except json.JSONDecodeError as e:
            print(f"JSON解析错误 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print("等待 3 秒后重试...")
                time.sleep(3)
            continue
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print("等待 5 秒后重试...")
                time.sleep(5)
            continue
    
    print("达到最大重试次数，返回空列表")
    return []

def validate_qa_pair(qa_pair):
    """验证问答对质量"""
    if not isinstance(qa_pair, dict):
        return False
    
    if 'question' not in qa_pair or 'answer' not in qa_pair:
        return False
    
    question = str(qa_pair['question']).strip()
    answer = str(qa_pair['answer']).strip()
    
    if len(question) < 5 or len(answer) < 5:
        return False
    
    if len(question) > 500 or len(answer) > 1000:
        return False
    
    return True

def generate_1000_qa_pairs():
    """生成1000个问答对"""
    if not OPENROUTER_API_KEY:
        print("错误：OPENROUTER_API_KEY 未设置")
        return
    
    chunks_dir = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/chunks"
    output_dir = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取文本块索引
    index_file = os.path.join(chunks_dir, "chunks_index.json")
    with open(index_file, 'r', encoding='utf-8') as f:
        chunks_index = json.load(f)
    
    chunks = chunks_index['chunks']
    print(f"找到 {len(chunks)} 个文本块")
    
    # 计算每个块需要生成的问答对数量
    target_total = 1000
    qa_per_chunk = max(45, target_total // len(chunks) + 5)  # 确保总数足够
    
    print(f"目标生成 {target_total} 个问答对")
    print(f"每个块将尝试生成 {qa_per_chunk} 个问答对")
    
    qa_file = os.path.join(output_dir, "qa_pairs_1000.jsonl")
    
    # 清空输出文件
    with open(qa_file, 'w', encoding='utf-8') as f:
        pass
    
    total_generated = 0
    
    for i, chunk_data in enumerate(tqdm(chunks, desc="生成问答对")):
        print(f"\n处理块 {i+1}/{len(chunks)}: {chunk_data['chapter']}")
        
        # 动态调整生成数量以确保达到目标
        remaining_chunks = len(chunks) - i
        remaining_target = target_total - total_generated
        current_target = max(qa_per_chunk, remaining_target // remaining_chunks + 10)
        
        messages = create_qa_prompt_messages(chunk_data['text'], current_target)
        qa_pairs = call_api_with_retry(messages)
        
        valid_pairs = []
        for qa in qa_pairs:
            if validate_qa_pair(qa):
                qa['chunk_id'] = chunk_data['chunk_id']
                qa['chapter'] = chunk_data['chapter']
                qa['chapter_num'] = chunk_data['chapter_num']
                valid_pairs.append(qa)
        
        # 写入文件
        with jsonlines.open(qa_file, mode='a') as writer:
            for qa in valid_pairs:
                writer.write(qa)
        
        total_generated += len(valid_pairs)
        print(f"本块生成 {len(valid_pairs)} 个有效问答对，总计 {total_generated} 个")
        
        # 如果已经达到目标，提前结束
        if total_generated >= target_total:
            print(f"已达到目标数量 {target_total}，停止生成")
            break
        
        # 避免API限制
        time.sleep(2)
    
    print(f"\n总共生成了 {total_generated} 个问答对")
    
    # 生成prompt-response格式
    prompt_response_file = os.path.join(output_dir, "prompt_response_1000.jsonl")
    
    with jsonlines.open(qa_file) as reader:
        with jsonlines.open(prompt_response_file, mode='w') as writer:
            for qa in reader:
                writer.write({
                    "prompt": f"根据经济学原理教材内容回答：{qa['question']}",
                    "response": qa['answer'],
                    "chunk_id": qa.get('chunk_id', ''),
                    "chapter": qa.get('chapter', ''),
                    "chapter_num": qa.get('chapter_num', '')
                })
    
    print(f"prompt-response格式文件已生成: {prompt_response_file}")
    print(f"问答对文件: {qa_file}")
    
    return total_generated

if __name__ == "__main__":
    total = generate_1000_qa_pairs()
    print(f"\n✅ 完成！总共生成了 {total} 个问答对") 