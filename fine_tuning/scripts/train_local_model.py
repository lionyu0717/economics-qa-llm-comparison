#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本地模型训练脚本
使用已经安装的GPT-2等本地模型进行微调
避免网络下载问题
"""

import os
import torch
import json
import jsonlines
from datasets import Dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model

def load_dataset(file_path, format_type="jsonl"):
    """加载数据集"""
    data = []
    if format_type == "jsonl":
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
    else:  # json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

def create_training_text_from_qwen_format(data):
    """从QWEN格式创建训练文本"""
    texts = []
    for item in data:
        # 提取问题和答案
        text = item.get("text", "")
        if "<|im_start|>user\n" in text and "<|im_start|>assistant\n" in text:
            question = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            answer = text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
            
            # 格式化为简单的QA格式
            formatted_text = f"问题：{question}\n答案：{answer}"
            texts.append(formatted_text)
    return texts

def create_training_text_from_alpaca_format(data):
    """从Alpaca格式创建训练文本"""
    texts = []
    for item in data:
        question = item.get("input", "")
        answer = item.get("output", "")
        
        # 格式化为简单的QA格式
        formatted_text = f"问题：{question}\n答案：{answer}"
        texts.append(formatted_text)
    return texts

def tokenize_function(examples, tokenizer, max_length=256):
    """tokenize数据"""
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 对于因果语言模型，labels就是input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def train_gpt2_model(data_texts, model_name="gpt2"):
    """训练GPT-2模型"""
    print(f"使用模型: {model_name}")
    output_dir = f"./fine_tuning/local/checkpoints_{model_name.replace('/', '_')}"
    
    # 加载tokenizer和模型
    print("加载tokenizer和模型...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # 设置pad_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 准备数据
    train_texts = data_texts[:80]  # 使用80个样本训练
    val_texts = data_texts[80:100]  # 使用20个样本验证
    
    print(f"训练样本数: {len(train_texts)}")
    print(f"验证样本数: {len(val_texts)}")
    
    # 创建Dataset
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        report_to=None,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def test_model(model_path, tokenizer_path):
    """测试微调后的模型"""
    print(f"测试模型: {model_path}")
    
    from peft import PeftModel
    
    # 加载模型
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 测试问题
    test_questions = [
        "什么是稀缺性？",
        "什么是机会成本？",
        "什么是市场均衡？"
    ]
    
    for question in test_questions:
        prompt = f"问题：{question}\n答案："
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        print(f"\n问题: {question}")
        print(f"回答: {answer}")

def main():
    print("🚀 本地模型微调训练")
    print("="*50)
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs("fine_tuning/local", exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    qwen_data = load_dataset("fine_tuning/data/qwen/train.jsonl", "jsonl")
    alpaca_data = load_dataset("fine_tuning/data/alpaca/train.json", "json")
    
    # 转换数据格式
    qwen_texts = create_training_text_from_qwen_format(qwen_data)
    alpaca_texts = create_training_text_from_alpaca_format(alpaca_data)
    
    print(f"QWEN格式文本: {len(qwen_texts)} 个")
    print(f"Alpaca格式文本: {len(alpaca_texts)} 个")
    
    # 使用QWEN数据训练第一个模型
    print("\n训练模型1 (基于QWEN数据)...")
    model1_path = train_gpt2_model(qwen_texts, "gpt2")
    print(f"模型1保存在: {model1_path}")
    
    # 使用Alpaca数据训练第二个模型
    print("\n训练模型2 (基于Alpaca数据)...")
    model2_path = train_gpt2_model(alpaca_texts, "gpt2")
    print(f"模型2保存在: {model2_path}")
    
    # 测试模型
    print("\n测试模型1...")
    test_model(model1_path, model1_path)
    
    print("\n测试模型2...")
    test_model(model2_path, model2_path)
    
    print("\n🎉 训练完成!")
    print("两个经济学问答模型已成功微调")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc() 