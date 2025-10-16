#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版LLAMA模型微调脚本
使用较小模型和更保守的设置
"""

import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model

def load_alpaca_dataset(file_path):
    """加载Alpaca格式的数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_alpaca_prompt(instruction, input_text, output=None):
    """格式化Alpaca提示"""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    if output:
        prompt += output
    
    return prompt

def tokenize_alpaca_function(examples, tokenizer, max_length=256):
    """tokenize Alpaca格式的数据"""
    # 构建输入文本
    texts = []
    for i in range(len(examples["instruction"])):
        full_prompt = format_alpaca_prompt(
            examples["instruction"][i],
            examples["input"][i],
            examples["output"][i]
        )
        texts.append(full_prompt)
    
    # tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 对于简化版，直接使用input_ids作为labels
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def main():
    # 配置 - 使用开源可用的小模型
    model_name = "microsoft/DialoGPT-medium"  # 使用DialoGPT作为替代
    output_dir = "./fine_tuning/llama/checkpoints"
    data_dir = "./fine_tuning/data/alpaca"
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA配置 - 更保守的设置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # 更小的rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"]  # DialoGPT的attention模块
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 加载数据
    print("加载训练数据...")
    train_data = load_alpaca_dataset(f"{data_dir}/train.json")
    val_data = load_alpaca_dataset(f"{data_dir}/val.json")
    
    # 只使用部分数据进行快速训练
    train_data = train_data[:100]  # 只用100个样本
    val_data = val_data[:20]      # 只用20个样本
    
    print(f"训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(val_data)}")
    
    # 转换为Dataset对象
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize数据
    print("Tokenizing数据...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_alpaca_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_alpaca_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 使用因果语言模型
        pad_to_multiple_of=8
    )
    
    # 训练参数 - 更保守的设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # 不使用wandb等
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # 避免多进程问题
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
    print("开始微调...")
    try:
        trainer.train()
        
        # 保存模型
        print("保存微调后的模型...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"微调完成！模型保存在: {output_dir}")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 