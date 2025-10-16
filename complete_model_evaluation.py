#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的经济学问答模型评估
包含标准的NLP评价指标，使用测试集进行性能评估
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from datetime import datetime

# NLP评估指标
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import jieba
    METRICS_AVAILABLE = True
except ImportError:
    print("警告: 部分评估指标库未安装，将使用基础指标")
    METRICS_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型配置 
        self.models_config = {
            "qwen3": {
                "base_model": "fine_tuning/models/qwen3-1.7b",
                "tuned_model": "fine_tuning/qwen3_economics_model",
                "display_name": "Qwen3-1.7B"
            },
            "gemma3": {
                "base_model": "google/gemma-3-1b-it", 
                "tuned_model": "fine_tuning/gemma3_economics_model",
                "display_name": "Gemma3-1B"
            }
        }
        
        # 测试数据路径
        self.test_data_path = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
        
        # 评估指标初始化
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            self.smoothing = SmoothingFunction().method1
        
    def load_test_data(self, max_samples=50):
        """加载测试数据"""
        logger.info("📚 加载测试数据...")
        
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        
        # 取最后的测试样本（和训练时的分割保持一致）
        total_samples = len(all_data)
        test_start = int(total_samples * 0.9)  # 前90%是训练+验证，后10%是测试
        test_samples = all_data[test_start:test_start + max_samples]
        
        for item in test_samples:
            test_data.append({
                "question": item["question"],
                "reference": item["answer"]
            })
        
        logger.info(f"加载了 {len(test_data)} 个测试样本")
        return test_data
    
    def load_model(self, model_name):
        """加载微调模型"""
        try:
            config = self.models_config[model_name]
            logger.info(f"🔄 加载 {config['display_name']} 模型...")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            # 加载LoRA适配器
            model = PeftModel.from_pretrained(base_model, config["tuned_model"])
            model.eval()
            
            logger.info(f"✅ {config['display_name']} 模型加载成功")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ 加载 {config['display_name']} 模型失败: {e}")
            return None, None
    
    def generate_answer_optimized(self, model, tokenizer, question, model_name):
        """使用优化参数生成回答（non-thinking模式）"""
        try:
            if model_name == "qwen3":
                # Qwen3 non-thinking模式  
                prompt = f"<|user|>\n你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。\n{question}\n<|assistant|>\n"
                
                # 官方推荐的non-thinking参数
                generation_params = {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "repetition_penalty": 1.3,  # 防止重复
                }
                
            else:  # gemma3
                # Gemma3使用聊天模板
                messages = [
                    {"role": "system", "content": "你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。"},
                    {"role": "user", "content": question}
                ]
                
                try:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    # 备用格式
                    prompt = f"<bos><start_of_turn>user\n你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。\n{question}<end_of_turn>\n<start_of_turn>model\n"
                
                generation_params = {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "repetition_penalty": 1.2,
                }
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成回答
            start_time = time.time()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_params)
            generation_time = time.time() - start_time
            
            # 解码输出
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.replace(prompt, "").strip()
            
            # 清理答案
            if "\n" in answer:
                lines = answer.split("\n")
                for line in lines:
                    if line.strip():
                        answer = line.strip()
                        break
            
            # 限制长度并移除可能的格式标记
            answer = answer.replace("<end_of_turn>", "").strip()
            if len(answer) > 300:
                answer = answer[:300] + "..."
            
            return answer, generation_time
            
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return f"[生成失败: {str(e)}]", 0
    
    def calculate_rouge_scores(self, reference, candidate):
        """计算ROUGE分数"""
        if not METRICS_AVAILABLE:
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure,
            }
        except:
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    def calculate_bleu_score(self, reference, candidate):
        """计算BLEU分数"""
        if not METRICS_AVAILABLE:
            return 0.0
        
        try:
            # 使用jieba分词
            ref_tokens = list(jieba.cut(reference))
            cand_tokens = list(jieba.cut(candidate))
            
            if len(cand_tokens) == 0:
                return 0.0
            
            return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def calculate_basic_similarity(self, reference, candidate):
        """计算基础相似度（词汇重叠）"""
        try:
            if METRICS_AVAILABLE:
                ref_words = set(jieba.cut(reference))
                cand_words = set(jieba.cut(candidate))
            else:
                ref_words = set(reference.split())
                cand_words = set(candidate.split())
            
            if len(ref_words) == 0 and len(cand_words) == 0:
                return 1.0
            
            intersection = len(ref_words & cand_words)
            union = len(ref_words | cand_words)
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def assess_answer_quality(self, question, reference, candidate):
        """评估回答质量"""
        try:
            # 1. 长度合理性
            ref_len = len(reference)
            cand_len = len(candidate)
            length_ratio = min(cand_len / max(ref_len, 1), 2.0) if cand_len > 0 else 0.0
            length_score = 1.0 - abs(1.0 - length_ratio) if length_ratio <= 2.0 else 0.0
            
            # 2. 关键词覆盖度
            if METRICS_AVAILABLE:
                question_keywords = set(jieba.cut(question))
                answer_keywords = set(jieba.cut(candidate))
            else:
                question_keywords = set(question.split())
                answer_keywords = set(candidate.split())
                
            keyword_coverage = len(question_keywords & answer_keywords) / max(len(question_keywords), 1)
            
            # 3. 流畅性检查
            fluency = 1.0
            if "生成失败" in candidate or len(candidate) < 5:
                fluency = 0.0
            elif len(candidate) > 50 and candidate.count("。") == 0:
                fluency = 0.7
            
            # 4. 重复检查
            repetition_score = 1.0
            sentences = candidate.split("。")
            if len(sentences) > 2:
                unique_sentences = set(sentences)
                repetition_score = len(unique_sentences) / len(sentences)
            
            overall_quality = (length_score + keyword_coverage + fluency + repetition_score) / 4
            
            return {
                'length_score': length_score,
                'keyword_coverage': keyword_coverage,
                'fluency': fluency,
                'repetition_score': repetition_score,
                'overall_quality': overall_quality
            }
        except:
            return {k: 0.0 for k in ['length_score', 'keyword_coverage', 'fluency', 'repetition_score', 'overall_quality']}
    
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
            reference = item["reference"]
            
            logger.info(f"🔸 处理问题 {i+1}/{len(test_data)}: {question[:50]}...")
            
            # 生成回答
            candidate, gen_time = self.generate_answer_optimized(model, tokenizer, question, model_name)
            total_time += gen_time
            
            # 计算各种评价指标
            rouge_scores = self.calculate_rouge_scores(reference, candidate)
            bleu_score = self.calculate_bleu_score(reference, candidate)
            basic_sim = self.calculate_basic_similarity(reference, candidate)
            quality_scores = self.assess_answer_quality(question, reference, candidate)
            
            result = {
                "question": question,
                "reference": reference,
                "candidate": candidate,
                "generation_time": gen_time,
                "answer_length": len(candidate),
                **rouge_scores,
                "bleu": bleu_score,
                "basic_similarity": basic_sim,
                **quality_scores
            }
            results.append(result)
            
            logger.info(f"   ⏱️ 耗时: {gen_time:.2f}s")
            logger.info(f"   📝 回答: {candidate[:100]}...")
        
        # 清理GPU内存
        del model, tokenizer
        torch.cuda.empty_cache()
        
        # 计算平均指标
        avg_metrics = {}
        numeric_fields = ['generation_time', 'answer_length', 'rouge1_f', 'rouge2_f', 'rougeL_f', 
                         'bleu', 'basic_similarity', 'overall_quality']
        
        for field in numeric_fields:
            values = [r[field] for r in results if isinstance(r[field], (int, float))]
            avg_metrics[f"avg_{field}"] = np.mean(values) if values else 0.0
        
        model_stats = {
            "model_name": self.models_config[model_name]['display_name'],
            "total_samples": len(test_data),
            "results": results,
            **avg_metrics
        }
        
        logger.info(f"✅ {self.models_config[model_name]['display_name']} 评估完成")
        logger.info(f"   📊 平均ROUGE-1: {avg_metrics['avg_rouge1_f']:.4f}")
        logger.info(f"   📊 平均BLEU: {avg_metrics['avg_bleu']:.4f}")
        logger.info(f"   📊 平均相似度: {avg_metrics['avg_basic_similarity']:.4f}")
        logger.info(f"   📊 平均质量评分: {avg_metrics['avg_overall_quality']:.4f}")
        
        return model_stats
    
    def generate_comprehensive_report(self, qwen_stats, gemma_stats):
        """生成综合评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_evaluation_report_{timestamp}.txt"
        
        report = []
        report.append("🎯 经济学问答模型综合评估报告")
        report.append("=" * 60)
        report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试样本数量: {qwen_stats['total_samples']}")
        report.append("")
        
        # 评价指标说明
        report.append("📊 评价指标说明:")
        report.append("-" * 40)
        report.append("• ROUGE-1: 词汇级别重叠度 (值越高越好)")
        report.append("• ROUGE-2: 双词组合重叠度 (值越高越好)")
        report.append("• ROUGE-L: 最长公共子序列重叠度 (值越高越好)")
        report.append("• BLEU: 机器翻译评估指标 (值越高越好)")
        report.append("• 基础相似度: 词汇重叠度 (值越高越好)")
        report.append("• 综合质量评分: 多维度质量评估 (值越高越好)")
        report.append("• 生成时间: 模型响应速度 (值越低越好)")
        report.append("")
        
        # 主要指标对比
        report.append("🏆 主要指标对比:")
        report.append("-" * 40)
        
        metrics = [
            ("ROUGE-1 F1", "avg_rouge1_f", "高"),
            ("ROUGE-2 F1", "avg_rouge2_f", "高"), 
            ("ROUGE-L F1", "avg_rougeL_f", "高"),
            ("BLEU评分", "avg_bleu", "高"),
            ("基础相似度", "avg_basic_similarity", "高"),
            ("综合质量", "avg_overall_quality", "高"),
            ("平均生成时间(s)", "avg_generation_time", "低"),
            ("平均回答长度", "avg_answer_length", "-")
        ]
        
        for metric_name, field, better in metrics:
            qwen_val = qwen_stats[field]
            gemma_val = gemma_stats[field]
            
            if better == "高":
                winner = "Qwen3" if qwen_val > gemma_val else "Gemma3"
                improvement = abs(qwen_val - gemma_val) / max(min(qwen_val, gemma_val), 0.001) * 100
            elif better == "低":
                winner = "Qwen3" if qwen_val < gemma_val else "Gemma3"
                improvement = abs(qwen_val - gemma_val) / max(max(qwen_val, gemma_val), 0.001) * 100
            else:
                winner = "-"
                improvement = 0
            
            report.append(f"{metric_name}:")
            report.append(f"  Qwen3:  {qwen_val:.4f}")
            report.append(f"  Gemma3: {gemma_val:.4f}")
            if winner != "-":
                report.append(f"  优胜者: {winner} (领先 {improvement:.1f}%)")
            report.append("")
        
        # 详细样本对比（显示前5个）
        report.append("🔍 详细样本对比 (前5个):")
        report.append("-" * 50)
        
        for i in range(min(5, len(qwen_stats["results"]))):
            qwen_result = qwen_stats["results"][i]
            gemma_result = gemma_stats["results"][i]
            
            report.append(f"\n问题 {i+1}: {qwen_result['question']}")
            report.append(f"参考答案: {qwen_result['reference'][:100]}...")
            report.append("")
            report.append(f"🟦 Qwen3回答 (ROUGE-1: {qwen_result['rouge1_f']:.3f}, BLEU: {qwen_result['bleu']:.3f}):")
            report.append(f"   {qwen_result['candidate'][:150]}...")
            report.append("")
            report.append(f"🟨 Gemma3回答 (ROUGE-1: {gemma_result['rouge1_f']:.3f}, BLEU: {gemma_result['bleu']:.3f}):")
            report.append(f"   {gemma_result['candidate'][:150]}...")
            report.append("-" * 40)
        
        # 评估结论
        report.append("\n🎉 评估结论:")
        report.append("-" * 40)
        
        qwen_wins = 0
        gemma_wins = 0
        
        for _, field, better in metrics[:6]:  # 前6个质量指标
            qwen_val = qwen_stats[field]
            gemma_val = gemma_stats[field]
            if better == "高" and qwen_val > gemma_val:
                qwen_wins += 1
            elif better == "高" and gemma_val > qwen_val:
                gemma_wins += 1
        
        if qwen_wins > gemma_wins:
            report.append("• 总体表现: Qwen3在大多数质量指标上表现更好")
        elif gemma_wins > qwen_wins:
            report.append("• 总体表现: Gemma3在大多数质量指标上表现更好")
        else:
            report.append("• 总体表现: 两个模型各有优势")
        
        speed_winner = "Qwen3" if qwen_stats["avg_generation_time"] < gemma_stats["avg_generation_time"] else "Gemma3"
        report.append(f"• 速度表现: {speed_winner} 响应更快")
        
        # 保存报告
        report_content = "\n".join(report)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 综合评估报告已保存: {report_file}")
        return report_file
    
    def run_evaluation(self, max_samples=30):
        """运行完整评估"""
        logger.info("🚀 开始综合模型评估...")
        print("\n🎯 经济学问答模型综合评估")
        print("包含标准NLP评价指标的测试集性能评估")
        print("=" * 60)
        
        try:
            # 加载测试数据
            test_data = self.load_test_data(max_samples)
            
            # 评估Qwen3
            print(f"\n🔵 评估 Qwen3-1.7B 模型 (Non-thinking模式)...")
            qwen_stats = self.evaluate_model("qwen3", test_data)
            if qwen_stats is None:
                logger.error("Qwen3模型评估失败")
                return
            
            # 评估Gemma3
            print(f"\n🟡 评估 Gemma3-1B 模型...")
            gemma_stats = self.evaluate_model("gemma3", test_data)
            if gemma_stats is None:
                logger.error("Gemma3模型评估失败")
                return
            
            # 生成综合报告
            print(f"\n📝 生成综合评估报告...")
            report_file = self.generate_comprehensive_report(qwen_stats, gemma_stats)
            
            logger.info("🎉 综合模型评估完成！")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ 评估过程失败: {e}")
            raise

def main():
    print("🎯 经济学问答模型综合评估")
    print("使用标准NLP评价指标对测试集进行性能评估")
    print("=" * 60)
    
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
    
    # 开始评估
    evaluator = ComprehensiveModelEvaluator()
    report_file = evaluator.run_evaluation(max_samples=30)  # 使用30个测试样本
    
    print(f"\n🎊 评估完成！查看详细报告: {report_file}")

if __name__ == "__main__":
    main() 