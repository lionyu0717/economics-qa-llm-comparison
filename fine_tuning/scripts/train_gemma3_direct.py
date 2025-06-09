#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用transformers库中的Gemma 3-1B模型进行微调
无需预下载，使用8bit量化节省显存
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gemma3EconomicsTrainer:
    def __init__(self, 
                 model_id="google/gemma-3-1b-it",
                 data_path="经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl",
                 output_dir="fine_tuning/gemma3_economics_model",
                 hf_token="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"):
        """
        初始化Gemma3经济学问答助手训练器
        
        Args:
            model_id: HuggingFace模型ID
            data_path: 训练数据路径
            output_dir: 模型输出目录
            hf_token: HuggingFace访问token
        """
        self.model_id = model_id
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"🚀 开始Gemma3经济学问答助手训练")
        logger.info(f"模型ID: {self.model_id}")
        logger.info(f"数据路径: {self.data_path}")
        logger.info(f"输出路径: {self.output_dir}")
    
    def load_model_and_tokenizer(self):
        """加载Gemma3模型和tokenizer，使用8bit量化"""
        logger.info("从transformers库直接加载Gemma3模型...")
        
        # 8bit量化配置，节省显存
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Tokenizer加载成功，词汇量: {len(self.tokenizer)}")
        
        # 加载模型
        self.model = Gemma3ForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            token=self.hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 获取模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型加载成功，参数量: {total_params:,}")
        
        return self.model, self.tokenizer
    
    def setup_lora(self):
        """配置LoRA参数"""
        logger.info("配置LoRA参数...")
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank
            lora_alpha=32,  # 缩放参数
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Gemma的attention模块
            bias="none",
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA配置完成")
        return self.model
    
    def load_and_prepare_data(self):
        """加载并预处理数据"""
        logger.info(f"加载数据: {self.data_path}")
        
        # 读取jsonl文件
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        logger.info(f"加载了 {len(data)} 条数据")
        
        # 数据分割
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")
        logger.info(f"测试集: {len(test_data)} 条")
        
        # 转换为Gemma格式的对话
        def format_gemma_conversation(question, answer):
            """格式化为Gemma的对话格式"""
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。"}]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": question}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                }
            ]
            
            # 使用tokenizer的chat template
            formatted_text = self.tokenizer.apply_chat_template(
                [messages],
                add_generation_prompt=False,
                tokenize=False
            )[0]  # 取第一个元素，因为输入是list of list
            
            return formatted_text
        
        # 格式化数据
        def process_data(data_list):
            processed = []
            for item in data_list:
                try:
                    formatted_text = format_gemma_conversation(item["question"], item["answer"])
                    processed.append({"text": formatted_text})
                except Exception as e:
                    logger.warning(f"格式化数据时出错: {e}")
                    continue
            return processed
        
        train_formatted = process_data(train_data)
        val_formatted = process_data(val_data)
        
        # 创建Dataset
        train_dataset = Dataset.from_list(train_formatted)
        val_dataset = Dataset.from_list(val_formatted)
        
        # Tokenize数据
        def tokenize_function(examples):
            # 正确的tokenization
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_overflowing_tokens=False,
                return_tensors=None,
            )
            
            # 设置labels等于input_ids（causal LM的标准做法）
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # 应用tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, val_dataset, test_data
    
    def train_model(self, train_dataset, val_dataset):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # 减小batch size，因为Gemma3可能更大
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # 增大accumulation来补偿小batch size
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            fp16=False,  # 使用8bit量化，不需要fp16
            bf16=True,   # 使用bf16
            dataloader_drop_last=False,
            report_to=None,
            remove_unused_columns=True,
            dataloader_num_workers=0,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # 创建trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # 使用新的参数名
        )
        
        # 开始训练
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        training_time = end_time - start_time
        logger.info(f"训练完成! 耗时: {training_time:.1f}秒")
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"模型已保存到: {self.output_dir}")
        
        return trainer
    
    def test_model(self, test_data, max_samples=5):
        """测试训练后的模型"""
        logger.info("测试模型性能...")
        
        # 切换到推理模式
        self.model.eval()
        
        test_samples = test_data[:max_samples]
        
        results = []
        for i, item in enumerate(test_samples):
            question = item["question"]
            expected_answer = item["answer"]
            
            # 构建输入
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。"}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}]
                    }
                ]
            ]
            
            # 生成回答
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取回答部分
                if "assistant" in generated_text:
                    generated_answer = generated_text.split("assistant")[-1].strip()
                else:
                    generated_answer = generated_text.strip()
                
                result = {
                    "question": question,
                    "expected": expected_answer,
                    "generated": generated_answer
                }
                results.append(result)
                
                logger.info(f"问题 {i+1}: {question}")
                logger.info(f"期望答案: {expected_answer[:100]}...")
                logger.info(f"生成答案: {generated_answer[:100]}...")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"生成答案时出错: {e}")
                continue
        
        return results
    
    def run_full_training(self):
        """运行完整的训练流程"""
        try:
            # 1. 加载模型
            self.load_model_and_tokenizer()
            
            # 2. 配置LoRA
            self.setup_lora()
            
            # 3. 准备数据
            train_dataset, val_dataset, test_data = self.load_and_prepare_data()
            
            # 4. 训练模型
            trainer = self.train_model(train_dataset, val_dataset)
            
            # 5. 测试模型
            results = self.test_model(test_data)
            
            logger.info("🎉 Gemma3经济学模型训练完成！")
            return results
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            raise

def main():
    print("🤖 Gemma3经济学问答助手训练")
    print("使用transformers库直接加载Gemma3-1B模型")
    
    # 检查文件
    required_files = [
        "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 缺少文件: {file_path}")
            return
    
    print("✅ 所有文件检查通过，开始训练...")
    
    # 创建训练器并运行
    trainer = Gemma3EconomicsTrainer()
    trainer.run_full_training()

if __name__ == "__main__":
    main() 