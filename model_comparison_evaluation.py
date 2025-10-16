#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比评估Qwen3和Gemma3两个经济学微调模型
在相同测试集上进行公平比较
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Gemma3ForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging
from collections import defaultdict
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = "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"
        
        # 模型配置
        self.models_config = {
            "qwen3": {
                "base_model": "fine_tuning/models/qwen3-1.7b",
                "tuned_model": "fine_tuning/qwen3_economics_model",
                "model_class": AutoModelForCausalLM,
                "display_name": "Qwen3-1.7B"
            },
            "gemma3": {
                "base_model": "google/gemma-3-1b-it",
                "tuned_model": "fine_tuning/gemma3_economics_model", 
                "model_class": Gemma3ForCausalLM,
                "display_name": "Gemma3-1B"
            }
        }
        
        # 测试数据路径
        self.test_data_path = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
        
    def load_test_data(self, max_samples=20):
        """加载测试数据"""
        logger.info("📚 加载测试数据...")
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        
        # 使用后面的数据作为测试集（前面用于训练）
        train_size = int(0.8 * len(all_data))
        val_size = int(0.1 * len(all_data))
        test_data = all_data[train_size + val_size:]
        
        # 选择测试样本
        test_samples = test_data[:max_samples]
        
        logger.info(f"加载了 {len(test_samples)} 个测试样本")
        return test_samples
    
    def load_model(self, model_name):
        """加载指定的模型"""
        config = self.models_config[model_name]
        logger.info(f"🔄 加载 {config['display_name']} 模型...")
        
        try:
            # 8bit量化配置
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            # 加载tokenizer
            if model_name == "qwen3":
                tokenizer = AutoTokenizer.from_pretrained(
                    config["base_model"], 
                    trust_remote_code=True
                )
            else:  # gemma3
                tokenizer = AutoTokenizer.from_pretrained(
                    config["base_model"], 
                    token=self.hf_token,
                    trust_remote_code=True
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            if model_name == "qwen3":
                base_model = config["model_class"].from_pretrained(
                    config["base_model"],
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:  # gemma3
                base_model = config["model_class"].from_pretrained(
                    config["base_model"],
                    quantization_config=quantization_config,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # 加载LoRA适配器
            model = PeftModel.from_pretrained(base_model, config["tuned_model"])
            model.eval()
            
            logger.info(f"✅ {config['display_name']} 模型加载成功")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ 加载 {config['display_name']} 模型失败: {e}")
            return None, None
    
    def generate_answer(self, model, tokenizer, question, model_name):
        """生成模型回答"""
        try:
            if model_name == "qwen3":
                # Qwen3格式
                prompt = f"<|user|>\n你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。\n{question}\n<|assistant|>\n"
                
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_response.replace(prompt, "").strip()
                
            else:  # gemma3
                # Gemma3格式
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
                
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取assistant的回答
                if "assistant" in full_response:
                    answer = full_response.split("assistant")[-1].strip()
                else:
                    answer = full_response.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return f"[生成失败: {str(e)}]"
    
    def evaluate_model(self, model_name, test_data):
        """评估单个模型"""
        logger.info(f"📊 评估 {self.models_config[model_name]['display_name']} 模型...")
        
        # 加载模型
        model, tokenizer = self.load_model(model_name)
        if model is None:
            return None
        
        results = []
        total_time = 0
        
        for i, item in enumerate(test_data):
            question = item["question"]
            expected_answer = item["answer"]
            
            logger.info(f"🔸 处理问题 {i+1}/{len(test_data)}: {question[:50]}...")
            
            # 生成回答并计时
            start_time = time.time()
            generated_answer = self.generate_answer(model, tokenizer, question, model_name)
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            
            result = {
                "question": question,
                "expected": expected_answer,
                "generated": generated_answer,
                "time": generation_time,
                "length": len(generated_answer)
            }
            results.append(result)
            
            logger.info(f"   ⏱️ 耗时: {generation_time:.2f}s")
            logger.info(f"   📝 回答: {generated_answer[:100]}...")
        
        # 清理GPU内存
        del model, tokenizer
        torch.cuda.empty_cache()
        
        # 计算统计信息
        avg_time = total_time / len(test_data)
        avg_length = np.mean([r["length"] for r in results])
        
        model_stats = {
            "model_name": self.models_config[model_name]['display_name'],
            "total_time": total_time,
            "avg_time": avg_time,
            "avg_length": avg_length,
            "results": results
        }
        
        logger.info(f"✅ {self.models_config[model_name]['display_name']} 评估完成")
        logger.info(f"   📊 平均耗时: {avg_time:.2f}s")
        logger.info(f"   📏 平均长度: {avg_length:.1f}字符")
        
        return model_stats
    
    def compare_answers(self, qwen_stats, gemma_stats):
        """比较两个模型的回答"""
        logger.info("🔍 分析回答质量...")
        
        comparison_results = []
        
        for i in range(len(qwen_stats["results"])):
            qwen_result = qwen_stats["results"][i]
            gemma_result = gemma_stats["results"][i]
            
            comparison = {
                "question": qwen_result["question"],
                "expected": qwen_result["expected"],
                "qwen_answer": qwen_result["generated"],
                "gemma_answer": gemma_result["generated"],
                "qwen_time": qwen_result["time"],
                "gemma_time": gemma_result["time"],
                "qwen_length": qwen_result["length"],
                "gemma_length": gemma_result["length"]
            }
            comparison_results.append(comparison)
        
        return comparison_results
    
    def generate_report(self, qwen_stats, gemma_stats, comparisons):
        """生成对比报告"""
        logger.info("📝 生成对比报告...")
        
        report = []
        report.append("🏆 Qwen3 vs Gemma3 经济学问答模型对比报告")
        report.append("=" * 60)
        report.append("")
        
        # 基本统计
        report.append("📊 基本性能统计:")
        report.append(f"├─ Qwen3-1.7B:")
        report.append(f"│  ├─ 平均响应时间: {qwen_stats['avg_time']:.2f}s")
        report.append(f"│  └─ 平均回答长度: {qwen_stats['avg_length']:.1f}字符")
        report.append(f"└─ Gemma3-1B:")
        report.append(f"   ├─ 平均响应时间: {gemma_stats['avg_time']:.2f}s")
        report.append(f"   └─ 平均回答长度: {gemma_stats['avg_length']:.1f}字符")
        report.append("")
        
        # 性能对比
        speed_winner = "Qwen3" if qwen_stats['avg_time'] < gemma_stats['avg_time'] else "Gemma3"
        speed_diff = abs(qwen_stats['avg_time'] - gemma_stats['avg_time'])
        
        report.append("⚡ 性能对比:")
        report.append(f"├─ 响应速度优胜: {speed_winner} (快 {speed_diff:.2f}s)")
        report.append("")
        
        # 详细对比
        report.append("🔍 详细回答对比:")
        report.append("-" * 60)
        
        for i, comp in enumerate(comparisons):  # 显示所有问题
            report.append(f"\n问题 {i+1}: {comp['question']}")
            report.append(f"期望答案: {comp['expected'][:100]}...")
            report.append("")
            report.append(f"🟦 Qwen3回答 ({comp['qwen_time']:.2f}s, {comp['qwen_length']}字符):")
            report.append(f"   {comp['qwen_answer'][:200]}...")
            report.append("")
            report.append(f"🟨 Gemma3回答 ({comp['gemma_time']:.2f}s, {comp['gemma_length']}字符):")
            report.append(f"   {comp['gemma_answer'][:200]}...")
            report.append("-" * 40)
        
        # 保存报告
        report_text = "\n".join(report)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"model_comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📄 报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("🏆 模型对比结果摘要")
        print("="*60)
        print(f"⚡ 响应速度: {speed_winner} 获胜 (快 {speed_diff:.2f}秒)")
        print(f"📏 回答长度: Qwen3平均{qwen_stats['avg_length']:.0f}字符, Gemma3平均{gemma_stats['avg_length']:.0f}字符")
        print(f"📊 详细对比请查看: {report_file}")
        print("="*60)
        
        return report_file
    
    def run_comparison(self, max_samples=10):
        """运行完整的模型对比"""
        logger.info("🚀 开始模型对比评估...")
        print("\n🤖 Qwen3 vs Gemma3 经济学问答模型对比")
        print("=" * 50)
        
        try:
            # 加载测试数据
            test_data = self.load_test_data(max_samples)
            
            # 评估Qwen3
            print(f"\n🔵 正在评估 Qwen3-1.7B 模型...")
            qwen_stats = self.evaluate_model("qwen3", test_data)
            if qwen_stats is None:
                logger.error("Qwen3模型评估失败")
                return
            
            # 评估Gemma3
            print(f"\n🟡 正在评估 Gemma3-1B 模型...")
            gemma_stats = self.evaluate_model("gemma3", test_data)
            if gemma_stats is None:
                logger.error("Gemma3模型评估失败")
                return
            
            # 对比分析
            print(f"\n🔍 对比分析结果...")
            comparisons = self.compare_answers(qwen_stats, gemma_stats)
            
            # 生成报告
            report_file = self.generate_report(qwen_stats, gemma_stats, comparisons)
            
            logger.info("🎉 模型对比评估完成！")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ 评估过程失败: {e}")
            raise

def main():
    print("🏆 经济学问答模型对比评估")
    print("对比 Qwen3-1.7B vs Gemma3-1B 微调模型")
    print("=" * 50)
    
    # 检查必要文件
    required_paths = [
        "fine_tuning/qwen3_economics_model",
        "fine_tuning/gemma3_economics_model",
        "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            print(f"❌ 缺少必要文件: {path}")
            return
    
    print("✅ 所有文件检查通过")
    
    # 开始对比
    comparator = ModelComparator()
    report_file = comparator.run_comparison(max_samples=10)  # 先用10个样本测试
    
    print(f"\n🎊 评估完成！查看详细报告: {report_file}")

if __name__ == "__main__":
    main() 