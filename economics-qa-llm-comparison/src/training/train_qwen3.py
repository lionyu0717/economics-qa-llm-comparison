#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地Qwen3-1.7B模型进行微调
基于经济学问答数据集训练经济学问答助手
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
    AutoModelForCausalLM,
    AutoTokenizer,
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

class Qwen3EconomicsTrainer:
    def __init__(self, 
                 model_path="fine_tuning/models/qwen3-1.7b",
                 data_path="经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl",
                 output_dir="fine_tuning/qwen3_economics_model"):
        """
        初始化Qwen3经济学微调器
        
        Args:
            model_path: 本地模型路径
            data_path: 训练数据路径
            output_dir: 输出目录
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查路径
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """
        加载本地模型和tokenizer
        """
        logger.info(f"从本地加载Qwen3模型: {self.model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer加载成功，词汇量: {self.tokenizer.vocab_size}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            logger.info(f"模型加载成功，参数量: {self.model.num_parameters():,}")
            
            # 配置LoRA
            self.setup_lora()
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def setup_lora(self):
        """
        配置LoRA参数
        """
        logger.info("配置LoRA参数...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA配置完成")
    
    def load_and_prepare_data(self):
        """
        加载并准备训练数据
        """
        logger.info(f"加载数据: {self.data_path}")
        
        # 读取JSONL文件
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        
        logger.info(f"加载了 {len(data)} 条数据")
        
        # 转换为训练格式
        formatted_data = []
        for item in data:
            # 构建对话格式的文本
            conversation = f"<|user|>\n{item['question']}\n<|assistant|>\n{item['answer']}<|endoftext|>"
            formatted_data.append({"text": conversation})
        
        # 分割数据
        train_size = int(0.8 * len(formatted_data))
        val_size = int(0.1 * len(formatted_data))
        
        train_data = formatted_data[:train_size]
        val_data = formatted_data[train_size:train_size + val_size]
        test_data = formatted_data[train_size + val_size:]
        
        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")
        logger.info(f"测试集: {len(test_data)} 条")
        
        # 创建Dataset
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize数据
        def tokenize_function(examples):
            # 正确的tokenization，确保返回input_ids和attention_mask
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,  # 启用padding
                max_length=512,
                return_overflowing_tokens=False,
                return_tensors=None,  # 不直接返回tensor，让DataCollator处理
            )
            # 对于语言模型，labels就是input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, val_dataset, test_data
    
    def train_model(self, train_dataset, val_dataset):
        """
        训练模型
        """
        logger.info("开始训练模型...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            eval_strategy="steps",  # 新版本使用eval_strategy而不是evaluation_strategy
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            report_to=None,  # 不上报到wandb等
            remove_unused_columns=True,  # 移除未使用的列
            dataloader_num_workers=0,  # 避免多进程问题
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 不使用masked language modeling
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 训练
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        logger.info(f"训练完成! 耗时: {end_time - start_time:.1f}秒")
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        logger.info(f"模型已保存到: {self.output_dir}")
        
        return trainer
    
    def test_model(self, test_data, max_samples=10):
        """
        测试训练后的模型
        """
        logger.info("测试模型性能...")
        
        # 切换到推理模式
        self.model.eval()
        
        test_samples = test_data[:max_samples]
        
        results = []
        for i, item in enumerate(test_samples):
            # test_data是原始数据格式，直接使用text字段
            text_content = item["text"]
            parts = text_content.replace("<|user|>\n", "").replace("<|endoftext|>", "").split("\n<|assistant|>\n")
            question = parts[0].strip()
            expected_answer = parts[1].strip() if len(parts) > 1 else ""
            
            # 生成回答
            input_text = f"<|user|>\n{question}\n<|assistant|>\n"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated_text.replace(input_text, "").strip()
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer
            })
            
            logger.info(f"\n问题 {i+1}: {question}")
            logger.info(f"期望回答: {expected_answer[:100]}...")
            logger.info(f"生成回答: {generated_answer[:100]}...")
            logger.info("-" * 80)
        
        # 保存测试结果
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {results_file}")
        
        return results
    
    def run_full_training(self):
        """
        运行完整的训练流程
        """
        logger.info("🚀 开始Qwen3经济学问答助手训练")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"数据路径: {self.data_path}")
        logger.info(f"输出路径: {self.output_dir}")
        
        try:
            # 1. 加载模型
            self.load_model_and_tokenizer()
            
            # 2. 准备数据
            train_dataset, val_dataset, test_data = self.load_and_prepare_data()
            
            # 3. 训练模型
            trainer = self.train_model(train_dataset, val_dataset)
            
            # 4. 测试模型
            results = self.test_model(test_data)
            
            # 5. 生成训练报告
            self.generate_training_report(trainer, results)
            
            logger.info("✅ 训练流程完成!")
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {str(e)}")
            raise
    
    def generate_training_report(self, trainer, test_results):
        """
        生成训练报告
        """
        report = {
            "training_time": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "training_args": trainer.args.to_dict(),
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", "N/A"),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
            "test_samples": len(test_results),
            "model_info": {
                "model_type": self.model.config.model_type,
                "num_parameters": self.model.num_parameters(),
                "vocab_size": self.tokenizer.vocab_size
            }
        }
        
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存到: {report_file}")

def main():
    """
    主函数
    """
    print("🤖 Qwen3经济学问答助手训练")
    print("使用本地Qwen3-1.7B模型和经济学问答数据集")
    print()
    
    # 检查模型是否存在
    model_path = Path("fine_tuning/models/qwen3-1.7b")
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        print("请先运行 download_models.py 下载模型")
        return
    
    # 检查数据是否存在
    data_path = Path("经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl")
    if not data_path.exists():
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    print("✅ 所有文件检查通过，开始训练...")
    
    trainer = Qwen3EconomicsTrainer()
    trainer.run_full_training()

if __name__ == "__main__":
    main() 