#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 OpenRouter API (通过 OpenAI 客户端) 生成经济学教材的问答对
使用方法：python generate_qa.py path/to/chunks_directory
"""

import os
import sys
import json
import random
import time
import argparse
import re
from tqdm import tqdm
import jsonlines
from openai import OpenAI # 修改: 导入 OpenAI
from dotenv import load_dotenv

# 加载环境变量，包括API密钥
load_dotenv()

# API 相关的全局变量
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "") # 可选
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "") # 可选

# 全局 OpenAI 客户端实例 (将在主逻辑中基于密钥初始化)
client = None

def create_qa_prompt_messages(text, num_questions=3):
    """创建用于生成问答对的提示消息列表"""
    system_prompt = """你是一位专业的经济学教授，正在为《经济学原理》教材的学生准备高质量的学习和测验材料。
请根据以下提供的教材文本片段，创建指定数量的、富有洞察力且多样化的问答对。

核心要求：
1.  **认知深度**：问题应涵盖从基本理解（如定义、识别）到应用（如根据原理解决问题、解释现象）乃至分析（如比较不同概念、评估论点、识别假设）等不同认知层次。
2.  **答案的准确性与来源**：所有问题的答案都必须能从提供的文本片段中直接、明确地找到。答案应简洁、精确，并忠实于原文内容。
3.  **教育价值**：问答对应有助于学生检验和巩固对经济学核心概念、原理及其应用的理解。
4.  **避免劣质问题**：避免过于简单、事实陈述性（除非是关键定义）、含糊不清或与经济学学习目标无关的问题。

提问多样性与具体性指引：
*   **多角度提问**：尝试从不同视角审视文本内容。例如，可以针对一个概念问其定义、重要性、优点、缺点、与其他概念的联系等。
*   **关注关键文本元素**：请特别注意并围绕以下元素设计问题：
    *   **核心概念和定义**：例如，"什么是机会成本？""解释一下边际分析。"
    *   **重要原理和法则**：例如，"需求法则阐述了什么？""规模报酬递增是如何产生的？"
    *   **关键术语**：针对文本中出现的经济学术语提问。
    *   **例子和例证**：要求学生解释例子如何阐释某个原理，或者根据原理举出新的例子（如果文本提供足够信息支持）。例如，"文本中的例子是如何说明贸易的好处的？"
    *   **原因与结果**：针对现象、政策或决策的原因和可能导致的结果提问。例如，"导致市场失灵的主要原因有哪些？""减税政策对消费者支出有何影响？"
    *   **比较与对比**：如果文本中出现多个相关概念，可以要求比较它们的异同。例如，"比较完全竞争市场和垄断市场的关键区别。"
    *   **假设与结论**：识别文本中模型或论证所依赖的假设，以及得出的主要结论。
*   **问题类型多样化**：鼓励使用多种提问方式，如：
    *   ""是什么……？"" ""请定义……？""
    *   ""为什么……？"" ""请解释……的原因。""
    *   ""如何……？"" ""……是如何运作的？""
    *   ""比较/对比 A 和 B。""
    *   ""举例说明……"" 或 ""文中的例子说明了什么？""
    *   ""如果发生……，根据文本内容推断可能会发生什么？"" (确保答案仍严格基于文本)
*   **保持具体**：问题应尽可能具体，指向文本中的特定信息点，而不是宽泛的概括。

输出格式要求：
*   **严格的JSON格式**：仅返回一个合法的JSON数组，其中每个元素是一个包含 "question" 和 "answer" 键的JSON对象。不要包含任何JSON数组之外的文字、解释或标记 (如 ```json ... ```)。
    例如:
    [
      {
        "question": "问题1的文本",
        "answer": "答案1的文本"
      },
      {
        "question": "问题2的文本",
        "answer": "答案2的文本"
      }
      // ... 更多问答对
    ]
"""
    user_prompt = f"""请仔细阅读并理解以上角色设定、核心要求和提问指引。
现在，请根据以下文本内容，生成 {num_questions} 个符合所有要求的问答对。

文本内容：
{text}

请确保只输出JSON数组。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def call_openrouter_api(messages, model="google/gemini-2.0-flash-001", temperature=0.2, max_retries=3, retry_delay=5):
    """调用 OpenRouter API 生成问答对"""
    global client
    if not client:
        print("错误：OpenAI 客户端未初始化。请确保 API 密钥已提供。")
        return None

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
                max_tokens=2048, # 根据需要调整
                # response_format={ "type": "json_object" }, # 如果模型和OpenRouter支持，可以强制JSON输出
                extra_headers=headers if headers else None
            )
            
            content = completion.choices[0].message.content.strip()
            
            try:
                # 尝试提取JSON, 有时模型可能仍在外部添加```json ... ```
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.IGNORECASE)
                if json_match:
                    content_to_parse = json_match.group(1)
                else:
                    # 如果没有找到 ```json ... ```, 直接尝试解析整个内容
                    # 也可能模型直接返回了纯JSON字符串
                    json_match_direct = re.search(r'^\s*\[[\s\S]*\]\s*$', content)
                    if json_match_direct:
                         content_to_parse = json_match_direct.group(0)
                    else:
                        # 如果看起来不像一个完整的JSON数组，打印警告，但仍尝试解析
                        print(f"警告：API返回的内容可能不是预期的JSON数组格式。内容预览: {content[:100]}...")
                        content_to_parse = content


                qa_pairs = json.loads(content_to_parse)
                return qa_pairs
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始API返回内容 (尝试 {attempt+1}): ", content[:500] + "..." if len(content) > 500 else content)
                if attempt == max_retries - 1:
                    print("达到最大JSON解析重试次数。")
                    return None
                # 等待后重试，可能是API返回格式暂时有问题
                print(f"等待 {retry_delay} 秒后重试JSON解析...")
                time.sleep(retry_delay)
            
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("达到最大API调用重试次数，放弃。")
                return None
    return None

def validate_qa_pair(qa_pair):
    """验证问答对的质量"""
    # 检查问题和答案是否存在且不为空
    if not isinstance(qa_pair, dict):
        return False
    
    if 'question' not in qa_pair or 'answer' not in qa_pair:
        return False
    
    question = qa_pair['question'].strip()
    answer = qa_pair['answer'].strip()
    
    # 检查问题和答案的长度
    if len(question) < 10 or len(answer) < 5:
        return False
    
    # 检查问题是否以问号结尾
    if not question.endswith('?') and not question.endswith('？'):
        return False
    
    return True

def generate_qa_for_chunk(chunk_data, output_file, model="google/gemini-2.0-flash-001", questions_per_chunk=3, temperature=0.2):
    """为单个文本块生成问答对"""
    chunk_text = chunk_data['text']
    chunk_info = f"章节: {chunk_data['chapter']} (编号: {chunk_data['chapter_num']}, 块编号: {chunk_data['chunk_num']})"
    
    print(f"为{chunk_info}生成问答对 (使用模型: {model})...")
    
    # 创建提示消息
    messages = create_qa_prompt_messages(chunk_text, questions_per_chunk)
    
    # 调用API
    qa_pairs = call_openrouter_api(messages, model, temperature)
    
    if not qa_pairs:
        print(f"未能为{chunk_info}生成任何问答对")
        return 0
    
    # 验证并过滤问答对
    valid_qa_pairs = []
    # API可能返回单个对象或列表，确保我们迭代的是列表
    if not isinstance(qa_pairs, list):
        print(f"警告：API返回的不是预期的列表格式，而是 {type(qa_pairs)}。尝试将其视为单个问答对的列表。")
        # 如果模型有时返回单个JSON对象而不是数组，这里可以尝试包装它
        # 但我们期望的是一个JSON数组，所以如果不是，可能是个问题
        # qa_pairs = [qa_pairs] # 取消注释此行如果模型有时返回单个对象
    
    for qa in qa_pairs: # 确保 qa_pairs 是可迭代的列表
        if validate_qa_pair(qa):
            # 添加元数据
            qa['chunk_id'] = chunk_data['chunk_id']
            qa['chapter'] = chunk_data['chapter']
            qa['chapter_num'] = chunk_data['chapter_num']
            valid_qa_pairs.append(qa)
        else:
            print(f"跳过无效问答对: {qa}")
    
    # 将有效问答对写入JSONL文件
    with jsonlines.open(output_file, mode='a') as writer:
        for qa in valid_qa_pairs:
            writer.write(qa)
    
    print(f"已为{chunk_info}添加 {len(valid_qa_pairs)} 个问答对")
    return len(valid_qa_pairs)

def convert_to_prompt_response_format(qa_pairs, output_file):
    """将问答对转换为prompt-response格式"""
    with jsonlines.open(output_file, mode='w') as writer:
        for qa in qa_pairs:
            prompt = f"文本：{qa.get('text', '')}\\n问题：{qa['question']}" # 确保 text 字段存在或有默认值
            response = qa['answer']
            writer.write({
                "prompt": prompt,
                "response": response,
                "chunk_id": qa.get('chunk_id', ''),
                "chapter": qa.get('chapter', ''),
                "chapter_num": qa.get('chapter_num', '')
            })
    print(f"已生成prompt-response格式文件: {output_file}")

def chunk_has_enough_text(chunk_data, min_words=100):
    """检查文本块是否包含足够的文本"""
    text = chunk_data.get('text', '')
    word_count = len(text.split())
    return word_count >= min_words

def split_dataset(qa_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_dir="dataset_split"):
    """将问答数据集拆分为训练、验证和测试集"""
    os.makedirs(output_dir, exist_ok=True)
    qa_pairs = []
    with jsonlines.open(qa_file) as reader:
        for obj in reader:
            qa_pairs.append(obj)
    
    random.shuffle(qa_pairs)
    total = len(qa_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = qa_pairs[:train_end]
    val_data = qa_pairs[train_end:val_end]
    test_data = qa_pairs[val_end:]
    
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    
    with jsonlines.open(train_file, mode='w') as writer:
        writer.write_all(train_data)
    with jsonlines.open(val_file, mode='w') as writer:
        writer.write_all(val_data)
    with jsonlines.open(test_file, mode='w') as writer:
        writer.write_all(test_data)
    
    print(f"数据集拆分完成:")
    print(f"- 训练集: {len(train_data)} 个样本 ({train_ratio*100:.1f}%)")
    print(f"- 验证集: {len(val_data)} 个样本 ({val_ratio*100:.1f}%)")
    print(f"- 测试集: {len(test_data)} 个样本 ({test_ratio*100:.1f}%)")
    
    return {
        "train": train_file, "val": val_file, "test": test_file,
        "train_count": len(train_data), "val_count": len(val_data), "test_count": len(test_data)
    }

def process_chunks_directory(chunks_dir, output_dir=None, model="google/gemini-2.0-flash-001", questions_per_chunk=3, 
                             sample_ratio=1.0, temperature=0.2, split_data=True):
    global client # 声明我们要修改全局客户端
    if not OPENROUTER_API_KEY: # 在函数开始时检查，以避免不必要的处理
        print("错误：OPENROUTER_API_KEY 未设置。请在 .env 文件中或作为环境变量提供。")
        sys.exit(1)

    # 初始化OpenAI客户端，现在我们确认密钥存在
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    if not os.path.isdir(chunks_dir):
        raise ValueError(f"指定的文本块目录不存在: {chunks_dir}")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(chunks_dir), "qa_dataset_openrouter") # 修改输出目录名
    os.makedirs(output_dir, exist_ok=True)
    
    index_file = os.path.join(chunks_dir, "chunks_index.json")
    if not os.path.exists(index_file):
        raise ValueError(f"未找到索引文件: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        chunks_index = json.load(f)
    
    all_chunks = chunks_index['chunks']
    print(f"找到 {len(all_chunks)} 个文本块")
    
    valid_chunks = [chunk for chunk in all_chunks if chunk_has_enough_text(chunk)]
    print(f"过滤后剩余 {len(valid_chunks)} 个有效文本块")
    
    if sample_ratio < 1.0:
        sample_size = max(1, int(len(valid_chunks) * sample_ratio))
        chunks_to_process = random.sample(valid_chunks, sample_size)
        print(f"将随机处理 {sample_size} 个文本块 (采样比率: {sample_ratio})")
    else:
        chunks_to_process = valid_chunks
        print(f"将处理所有 {len(valid_chunks)} 个文本块")
    
    qa_file = os.path.join(output_dir, "qa_pairs_openrouter.jsonl") # 修改文件名
    with open(qa_file, 'w', encoding='utf-8') as f: # 确保文件存在且为空
        pass 
    
    total_qa_pairs = 0
    for chunk_data in tqdm(chunks_to_process, desc="生成问答对 (OpenRouter)"):
        num_generated = generate_qa_for_chunk(chunk_data, qa_file, model, questions_per_chunk, temperature)
        total_qa_pairs += num_generated
        time.sleep(1) # OpenRouter 可能有速率限制，保持一个小的延迟
    
    print(f"总共生成了 {total_qa_pairs} 个问答对，保存到: {qa_file}")
    
    prompt_response_file = os.path.join(output_dir, "prompt_response_openrouter.jsonl") # 修改文件名
    
    qa_pairs_for_conversion = []
    with jsonlines.open(qa_file) as reader:
        for obj in reader:
            qa_pairs_for_conversion.append(obj)
    
    convert_to_prompt_response_format(qa_pairs_for_conversion, prompt_response_file)
    
    split_info_val = None
    if split_data:
        split_info_val = split_dataset(prompt_response_file, output_dir=os.path.join(output_dir, "split_openrouter")) # 修改目录名
        print("数据集已拆分完成。")
    
    return {
        "qa_file": qa_file,
        "prompt_response_file": prompt_response_file,
        "total_qa_pairs": total_qa_pairs,
        "split_info": split_info_val
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用OpenRouter API生成问答对")
    parser.add_argument("chunks_dir", help="包含文本块的目录路径")
    parser.add_argument("--output", "-o", help="输出目录", default=None)
    parser.add_argument("--model", "-m", help="OpenRouter上使用的模型名称", default="google/gemini-2.0-flash-001")
    parser.add_argument("--questions", "-q", type=int, help="每个块生成的问题数量", default=3)
    parser.add_argument("--sample", "-s", type=float, help="处理的文本块比例 (0-1)", default=1.0)
    parser.add_argument("--temperature", "-t", type=float, help="生成的随机性 (0-1)", default=0.2)
    parser.add_argument("--no-split", action="store_false", dest="split_data", help="不拆分数据集")
    # API 密钥现在从环境变量 OPENROUTER_API_KEY 读取，不再通过命令行参数传递给此脚本
    # 如果需要，可以添加一个 --key 参数，但通常 .env 更方便管理

    args = parser.parse_args()

    # 检查 OPENROUTER_API_KEY 是否已在环境变量中设置 (通过 .env 或直接导出)
    # client 初始化移至 process_chunks_directory 内部，确保KEY已加载
    if not OPENROUTER_API_KEY:
        print("错误：OPENROUTER_API_KEY 环境变量未设置。请在 .env 文件中定义或导出该变量。")
        print("例如： export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)
    
    try:
        result = process_chunks_directory(
            args.chunks_dir, 
            args.output, 
            args.model, 
            args.questions, 
            args.sample,
            args.temperature,
            args.split_data
        )
        
        print("\n处理完成！")
        print(f"问答对文件: {result['qa_file']}")
        print(f"Prompt-Response格式文件: {result['prompt_response_file']}")
        print(f"总问答对数量: {result['total_qa_pairs']}")
        
        if result['split_info']:
            print("\n数据集拆分:")
            print(f"- 训练集: {result['split_info']['train']} ({result['split_info']['train_count']} 个样本)")
            print(f"- 验证集: {result['split_info']['val']} ({result['split_info']['val_count']} 个样本)")
            print(f"- 测试集: {result['split_info']['test']} ({result['split_info']['test_count']} 个样本)")
        
    except Exception as e:
        print(f"在 generate_qa.py 主流程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 