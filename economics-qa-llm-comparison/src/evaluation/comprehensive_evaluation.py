#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估Qwen3经济学模型微调效果
包含多种评估指标和对比分析
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 安装必要的包
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    pass

class EconomicsModelEvaluator:
    def __init__(self):
        self.base_model_path = "fine_tuning/models/qwen3-1.7b"
        self.tuned_model_path = "fine_tuning/qwen3_economics_model"
        self.data_path = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        self.smoothing = SmoothingFunction().method1
        
        # 评估结果
        self.results = {
            'base_model': defaultdict(list),
            'tuned_model': defaultdict(list)
        }
    
    def load_test_data(self, max_samples=50):
        """加载测试数据"""
        print("📚 加载测试数据...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # 使用后20%作为测试集
        test_start = int(len(data) * 0.8)
        test_data = data[test_start:test_start + max_samples]
        
        processed_data = []
        for item in test_data:
            # 原始数据格式是 {"question": "...", "answer": "..."}
            question = item["question"]
            reference_answer = item["answer"]
            processed_data.append({
                "question": question,
                "reference": reference_answer
            })
        
        print(f"✅ 加载了 {len(processed_data)} 个测试样本")
        return processed_data
    
    def load_models(self):
        """加载原始模型和微调模型"""
        print("🤖 加载模型...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载原始模型
        print("📦 加载原始模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        base_model.eval()
        
        # 加载微调模型
        print("🎯 加载微调模型...")
        tuned_base = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tuned_model = PeftModel.from_pretrained(tuned_base, self.tuned_model_path)
        tuned_model.eval()
        
        return tokenizer, base_model, tuned_model
    
    def generate_answer(self, model, tokenizer, question, max_new_tokens=150):
        """生成回答"""
        input_text = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.replace(input_text, "").strip()
            
            # 清理答案，移除重复或无关内容
            if "\n" in answer:
                answer = answer.split("\n")[0]
            
            return answer[:300]  # 限制长度
            
        except Exception as e:
            return f"生成失败: {str(e)}"
    
    def calculate_rouge_scores(self, reference, candidate):
        """计算ROUGE分数"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def calculate_bleu_score(self, reference, candidate):
        """计算BLEU分数"""
        try:
            # 使用jieba分词
            ref_tokens = list(jieba.cut(reference))
            cand_tokens = list(jieba.cut(candidate))
            
            if len(cand_tokens) == 0:
                return 0.0
            
            return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, reference, candidate):
        """计算语义相似度（简化版本）"""
        # 计算词汇重叠度
        ref_words = set(jieba.cut(reference))
        cand_words = set(jieba.cut(candidate))
        
        if len(ref_words) == 0 and len(cand_words) == 0:
            return 1.0
        
        intersection = len(ref_words & cand_words)
        union = len(ref_words | cand_words)
        
        return intersection / union if union > 0 else 0.0
    
    def assess_answer_quality(self, question, reference, candidate):
        """评估回答质量（多维度）"""
        
        # 1. 长度合理性 (0-1)
        ref_len = len(reference)
        cand_len = len(candidate)
        length_ratio = min(cand_len / max(ref_len, 1), 1.0) if cand_len > 0 else 0.0
        
        # 2. 是否回答了问题（检查关键词）
        question_keywords = set(jieba.cut(question))
        answer_keywords = set(jieba.cut(candidate))
        keyword_coverage = len(question_keywords & answer_keywords) / max(len(question_keywords), 1)
        
        # 3. 流畅性（简化评估：检查是否有明显错误）
        fluency = 1.0
        if "生成失败" in candidate or len(candidate) < 10:
            fluency = 0.0
        elif candidate.count("。") == 0 and len(candidate) > 50:  # 长文本没有句号
            fluency = 0.7
        
        return {
            'length_appropriateness': length_ratio,
            'keyword_coverage': keyword_coverage,
            'fluency': fluency,
            'overall_quality': (length_ratio + keyword_coverage + fluency) / 3
        }
    
    def evaluate_model(self, model, tokenizer, test_data, model_name):
        """评估单个模型"""
        print(f"📊 评估{model_name}...")
        
        all_results = []
        
        for i, item in enumerate(test_data):
            if i % 10 == 0:
                print(f"  进度: {i+1}/{len(test_data)}")
            
            question = item["question"]
            reference = item["reference"]
            
            # 生成回答
            candidate = self.generate_answer(model, tokenizer, question)
            
            # 计算各种指标
            rouge_scores = self.calculate_rouge_scores(reference, candidate)
            bleu_score = self.calculate_bleu_score(reference, candidate)
            semantic_sim = self.calculate_semantic_similarity(reference, candidate)
            quality_scores = self.assess_answer_quality(question, reference, candidate)
            
            result = {
                'question': question,
                'reference': reference,
                'candidate': candidate,
                'rouge1_f': rouge_scores['rouge1_f'],
                'rouge2_f': rouge_scores['rouge2_f'],
                'rougeL_f': rouge_scores['rougeL_f'],
                'bleu': bleu_score,
                'semantic_similarity': semantic_sim,
                **quality_scores
            }
            
            all_results.append(result)
        
        return all_results
    
    def compare_models(self, base_results, tuned_results):
        """对比两个模型的性能"""
        print("🔍 对比分析...")
        
        # 计算平均指标
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'semantic_similarity', 'overall_quality']
        
        comparison = {}
        for metric in metrics:
            base_avg = np.mean([r[metric] for r in base_results])
            tuned_avg = np.mean([r[metric] for r in tuned_results])
            improvement = (tuned_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0
            
            comparison[metric] = {
                'base_model': base_avg,
                'tuned_model': tuned_avg,
                'improvement_pct': improvement
            }
        
        return comparison
    
    def save_detailed_results(self, base_results, tuned_results, comparison):
        """保存详细结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_results = {
            'timestamp': timestamp,
            'base_model_results': base_results,
            'tuned_model_results': tuned_results,
            'comparison': comparison
        }
        
        output_file = f"evaluation_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果已保存到: {output_file}")
        
        # 创建简要报告
        self.create_summary_report(comparison, output_file.replace('.json', '_summary.txt'))
        
        return output_file
    
    def create_summary_report(self, comparison, output_file):
        """创建简要评估报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎯 Qwen3经济学模型微调效果评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 主要指标对比:\n")
            f.write("-" * 40 + "\n")
            
            metrics_chinese = {
                'rouge1_f': 'ROUGE-1 (词汇重叠)',
                'rouge2_f': 'ROUGE-2 (短语重叠)', 
                'rougeL_f': 'ROUGE-L (最长公共子序列)',
                'bleu': 'BLEU (翻译评估)',
                'semantic_similarity': '语义相似度',
                'overall_quality': '综合质量评分'
            }
            
            for metric, data in comparison.items():
                if metric in metrics_chinese:
                    name = metrics_chinese[metric]
                    f.write(f"{name}:\n")
                    f.write(f"  原始模型: {data['base_model']:.4f}\n")
                    f.write(f"  微调模型: {data['tuned_model']:.4f}\n")
                    f.write(f"  提升幅度: {data['improvement_pct']:+.2f}%\n\n")
            
            # 评估结论
            avg_improvement = np.mean([data['improvement_pct'] for data in comparison.values()])
            f.write("🎉 评估结论:\n")
            f.write("-" * 40 + "\n")
            
            if avg_improvement > 10:
                f.write("✅ 微调效果显著！模型在经济学问答方面有明显提升。\n")
            elif avg_improvement > 5:
                f.write("✅ 微调效果良好，模型性能有所改善。\n")
            elif avg_improvement > 0:
                f.write("⚠️ 微调效果一般，有轻微提升。\n")
            else:
                f.write("❌ 微调效果不明显，可能需要调整训练策略。\n")
            
            f.write(f"平均提升幅度: {avg_improvement:+.2f}%\n")
        
        print(f"📋 评估报告已保存到: {output_file}")
    
    def run_comprehensive_evaluation(self, max_samples=30):
        """运行完整评估"""
        print("🚀 开始综合评估...")
        print("=" * 50)
        
        # 1. 加载测试数据
        test_data = self.load_test_data(max_samples)
        
        # 2. 加载模型
        tokenizer, base_model, tuned_model = self.load_models()
        
        # 3. 评估原始模型
        base_results = self.evaluate_model(base_model, tokenizer, test_data, "原始模型")
        
        # 4. 评估微调模型
        tuned_results = self.evaluate_model(tuned_model, tokenizer, test_data, "微调模型")
        
        # 5. 对比分析
        comparison = self.compare_models(base_results, tuned_results)
        
        # 6. 保存结果
        output_file = self.save_detailed_results(base_results, tuned_results, comparison)
        
        # 7. 显示摘要
        self.print_summary(comparison)
        
        return output_file
    
    def print_summary(self, comparison):
        """打印评估摘要"""
        print("\n🎯 评估结果摘要")
        print("=" * 40)
        
        for metric, data in comparison.items():
            if metric in ['rouge1_f', 'bleu', 'overall_quality']:
                print(f"{metric.upper()}:")
                print(f"  原始: {data['base_model']:.4f}")
                print(f"  微调: {data['tuned_model']:.4f}")
                print(f"  提升: {data['improvement_pct']:+.2f}%")
                print()

def main():
    print("🎯 Qwen3经济学模型综合评估")
    print("=" * 50)
    
    evaluator = EconomicsModelEvaluator()
    
    try:
        output_file = evaluator.run_comprehensive_evaluation(max_samples=20)  # 先用20个样本测试
        print(f"\n🎉 评估完成！结果文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 