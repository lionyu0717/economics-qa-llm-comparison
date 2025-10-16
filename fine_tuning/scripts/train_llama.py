#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLAMA模型微调脚本
使用LoRA进行高效微调，采用Alpaca格式
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
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

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

def tokenize_alpaca_function(examples, tokenizer, max_length=512):
    """tokenize Alpaca格式的数据"""
    # 构建输入文本
    inputs = []
    for i in range(len(examples["instruction"])):
        full_prompt = format_alpaca_prompt(
            examples["instruction"][i],
            examples["input"][i],
            examples["output"][i]
        )
        inputs.append(full_prompt)
    
    # tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 创建labels
    labels = []
    for i, input_text in enumerate(inputs):
        # 找到"### Response:"的位置
        response_start = input_text.find("### Response:\n") + len("### Response:\n")
        prompt_part = input_text[:response_start]
        
        # tokenize prompt部分
        prompt_tokens = tokenizer(prompt_part, add_special_tokens=False, return_tensors=None)
        prompt_length = len(prompt_tokens["input_ids"])
        
        # 创建label：prompt部分设为-100，response部分保留
        label = model_inputs["input_ids"][i].copy()
        for j in range(prompt_length):
            if j < len(label):
                label[j] = -100
        
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs

def main():
    # 配置
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # 使用3B模型
    output_dir = "./fine_tuning/llama/checkpoints"
    data_dir = "./fine_tuning/data/alpaca"
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # 8bit量化节省显存
    )
    
    # 准备模型用于8bit训练
    model = prepare_model_for_kbit_training(model)
    
    # LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 加载数据
    print("加载训练数据...")
    train_data = load_alpaca_dataset(f"{data_dir}/train.json")
    val_data = load_alpaca_dataset(f"{data_dir}/val.json")
    
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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # 不使用wandb等
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
    trainer.train()
    
    # 保存模型
    print("保存微调后的模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"微调完成！模型保存在: {output_dir}")

if __name__ == "__main__":
    main() 