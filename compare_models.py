#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比微调前后模型的回答质量
更直观地评估微调效果
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

class ModelComparator:
    def __init__(self):
        self.base_model_path = "fine_tuning/models/qwen3-1.7b"
        self.tuned_model_path = "fine_tuning/qwen3_economics_model"
        
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
        
        print("✅ 模型加载完成!")
        return tokenizer, base_model, tuned_model
    
    def generate_answer(self, model, tokenizer, question, model_name=""):
        """生成回答"""
        input_text = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        try:
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.replace(input_text, "").strip()
            
            # 清理答案
            if "\n" in answer:
                answer = answer.split("\n")[0]
            
            return answer[:300], generation_time
            
        except Exception as e:
            return f"生成失败: {str(e)}", 0
    
    def get_test_questions(self):
        """获取测试问题"""
        return [
            "什么是稀缺性？",
            "请解释供给与需求的关系", 
            "什么是机会成本？",
            "请说明价格弹性的概念",
            "什么是市场失灵？",
            "解释通货膨胀的含义",
            "什么是比较优势？",
            "请说明生产可能性边界的概念",
            "什么是边际效用？",
            "解释经济学中的效率概念"
        ]
    
    def compare_models(self):
        """对比两个模型"""
        print("🎯 微调前后模型对比评估")
        print("=" * 60)
        
        # 加载模型
        tokenizer, base_model, tuned_model = self.load_models()
        
        # 获取测试问题
        questions = self.get_test_questions()
        
        print(f"\n📝 开始对比测试 ({len(questions)} 个问题)...")
        print("=" * 60)
        
        results = []
        
        for i, question in enumerate(questions):
            print(f"\n❓ 问题 {i+1}: {question}")
            print("=" * 50)
            
            # 原始模型回答
            print("📦 原始模型回答:")
            base_answer, base_time = self.generate_answer(base_model, tokenizer, question, "原始")
            print(f"💭 {base_answer}")
            print(f"⏱️ 生成时间: {base_time:.2f}秒")
            
            print("\n" + "-" * 50)
            
            # 微调模型回答
            print("🎯 微调模型回答:")
            tuned_answer, tuned_time = self.generate_answer(tuned_model, tokenizer, question, "微调")
            print(f"💡 {tuned_answer}")
            print(f"⏱️ 生成时间: {tuned_time:.2f}秒")
            
            # 保存结果
            results.append({
                "question": question,
                "base_answer": base_answer,
                "tuned_answer": tuned_answer,
                "base_time": base_time,
                "tuned_time": tuned_time
            })
            
            print("\n" + "="*60)
        
        # 保存对比结果
        self.save_comparison_results(results)
        
        # 显示总结
        self.print_summary(results)
        
        return results
    
    def save_comparison_results(self, results):
        """保存对比结果"""
        output_file = "model_comparison_results.json"
        
        comparison_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(results),
            "results": results,
            "summary": self.calculate_summary(results)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 对比结果已保存到: {output_file}")
        
        # 创建可读的报告
        self.create_readable_report(results)
    
    def calculate_summary(self, results):
        """计算总结统计"""
        base_times = [r["base_time"] for r in results if r["base_time"] > 0]
        tuned_times = [r["tuned_time"] for r in results if r["tuned_time"] > 0]
        
        base_avg_len = sum(len(r["base_answer"]) for r in results) / len(results)
        tuned_avg_len = sum(len(r["tuned_answer"]) for r in results) / len(results)
        
        return {
            "base_avg_time": sum(base_times) / len(base_times) if base_times else 0,
            "tuned_avg_time": sum(tuned_times) / len(tuned_times) if tuned_times else 0,
            "base_avg_length": base_avg_len,
            "tuned_avg_length": tuned_avg_len
        }
    
    def create_readable_report(self, results):
        """创建可读的对比报告"""
        report_file = "model_comparison_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎯 Qwen3 经济学模型微调前后对比报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试问题数量: {len(results)}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"问题 {i+1}: {result['question']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"原始模型: {result['base_answer']}\n\n")
                f.write(f"微调模型: {result['tuned_answer']}\n\n")
                f.write("=" * 60 + "\n\n")
            
            # 总结部分
            summary = self.calculate_summary(results)
            f.write("📊 对比总结:\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始模型平均回答时间: {summary['base_avg_time']:.2f}秒\n")
            f.write(f"微调模型平均回答时间: {summary['tuned_avg_time']:.2f}秒\n")
            f.write(f"原始模型平均回答长度: {summary['base_avg_length']:.1f}字符\n")
            f.write(f"微调模型平均回答长度: {summary['tuned_avg_length']:.1f}字符\n")
        
        print(f"📋 可读报告已保存到: {report_file}")
    
    def print_summary(self, results):
        """打印总结"""
        summary = self.calculate_summary(results)
        
        print("\n🎉 对比评估完成!")
        print("=" * 40)
        print("📊 快速总结:")
        print(f"  测试问题: {len(results)} 个")
        print(f"  原始模型平均用时: {summary['base_avg_time']:.2f}秒")
        print(f"  微调模型平均用时: {summary['tuned_avg_time']:.2f}秒")
        print(f"  原始模型平均回答长度: {summary['base_avg_length']:.1f}字符")
        print(f"  微调模型平均回答长度: {summary['tuned_avg_length']:.1f}字符")
        
        print("\n💡 评估建议:")
        print("  1. 查看 model_comparison_report.txt 了解详细对比")
        print("  2. 关注微调模型是否更专业、准确")
        print("  3. 观察回答的连贯性和相关性")
        print("  4. 检查是否有明显的过拟合现象")
        
        # 给出简单评价
        if summary['tuned_avg_length'] > summary['base_avg_length'] * 1.2:
            print("\n✅ 微调模型回答更详细，可能学到了更多经济学知识")
        elif summary['tuned_avg_length'] < summary['base_avg_length'] * 0.8:
            print("\n⚠️ 微调模型回答较短，可能需要调整训练参数")
        else:
            print("\n🔍 需要人工评估回答质量来判断微调效果")

def main():
    print("🎯 模型对比评估工具")
    print("对比微调前后模型在经济学问题上的表现")
    print("=" * 50)
    
    comparator = ModelComparator()
    
    try:
        results = comparator.compare_models()
        print(f"\n🎉 评估完成！共测试了 {len(results)} 个问题")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 