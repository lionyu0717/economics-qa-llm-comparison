#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐个测试模型的对比脚本
避免显存不足问题
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import gc

class SequentialModelComparator:
    def __init__(self):
        self.base_model_path = "fine_tuning/models/qwen3-1.7b"
        self.tuned_model_path = "fine_tuning/qwen3_economics_model"
        
    def clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_base_model(self):
        """加载原始模型"""
        print("📦 加载原始模型...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        return tokenizer, model
    
    def load_tuned_model(self):
        """加载微调模型"""
        print("🎯 加载微调模型...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tuned_model = PeftModel.from_pretrained(base_model, self.tuned_model_path)
        tuned_model.eval()
        
        return tokenizer, tuned_model
    
    def generate_answer(self, model, tokenizer, question):
        """生成回答"""
        input_text = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        try:
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
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
            
            return answer[:250], generation_time
            
        except Exception as e:
            return f"生成失败: {str(e)}", 0
    
    def get_test_questions(self):
        """获取测试问题"""
        return [
            "什么是稀缺性？",
            "请解释供给与需求的关系", 
            "什么是机会成本？",
            "什么是市场失灵？",
            "解释通货膨胀的含义",
            "什么是比较优势？"
        ]
    
    def test_model(self, model_type="base"):
        """测试单个模型"""
        questions = self.get_test_questions()
        results = []
        
        if model_type == "base":
            tokenizer, model = self.load_base_model()
            print(f"✅ 原始模型加载完成!")
        else:
            tokenizer, model = self.load_tuned_model()
            print(f"✅ 微调模型加载完成!")
        
        print(f"\n📝 开始测试 {model_type} 模型...")
        print("=" * 50)
        
        for i, question in enumerate(questions):
            print(f"\n❓ 问题 {i+1}: {question}")
            print("-" * 40)
            
            answer, gen_time = self.generate_answer(model, tokenizer, question)
            print(f"💡 回答: {answer}")
            print(f"⏱️ 时间: {gen_time:.2f}秒")
            
            results.append({
                "question": question,
                "answer": answer,
                "time": gen_time
            })
        
        # 清理内存
        del model, tokenizer
        self.clear_gpu_memory()
        
        return results
    
    def compare_models(self):
        """对比两个模型"""
        print("🎯 逐个测试模型对比")
        print("=" * 50)
        
        # 测试原始模型
        print("\n🔄 第1阶段: 测试原始模型")
        base_results = self.test_model("base")
        
        print("\n⏸️ 等待3秒清理内存...")
        time.sleep(3)
        
        # 测试微调模型
        print("\n🔄 第2阶段: 测试微调模型")
        tuned_results = self.test_model("tuned")
        
        # 生成对比报告
        self.create_comparison_report(base_results, tuned_results)
        
        return base_results, tuned_results
    
    def create_comparison_report(self, base_results, tuned_results):
        """创建对比报告"""
        print("\n📋 生成对比报告...")
        
        # 计算统计信息
        base_avg_time = sum(r["time"] for r in base_results) / len(base_results)
        tuned_avg_time = sum(r["time"] for r in tuned_results) / len(tuned_results)
        base_avg_len = sum(len(r["answer"]) for r in base_results) / len(base_results)
        tuned_avg_len = sum(len(r["answer"]) for r in tuned_results) / len(tuned_results)
        
        # 保存JSON结果
        comparison_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_results": base_results,
            "tuned_results": tuned_results,
            "summary": {
                "base_avg_time": base_avg_time,
                "tuned_avg_time": tuned_avg_time,
                "base_avg_length": base_avg_len,
                "tuned_avg_length": tuned_avg_len
            }
        }
        
        with open("sequential_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        # 创建可读报告
        with open("model_comparison_sequential.txt", 'w', encoding='utf-8') as f:
            f.write("🎯 Qwen3 经济学模型微调前后对比报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试问题数量: {len(base_results)}\n\n")
            
            for i, (base, tuned) in enumerate(zip(base_results, tuned_results)):
                f.write(f"问题 {i+1}: {base['question']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"原始模型 ({base['time']:.2f}s): {base['answer']}\n\n")
                f.write(f"微调模型 ({tuned['time']:.2f}s): {tuned['answer']}\n\n")
                f.write("=" * 60 + "\n\n")
            
            # 统计总结
            f.write("📊 对比总结:\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始模型平均回答时间: {base_avg_time:.2f}秒\n")
            f.write(f"微调模型平均回答时间: {tuned_avg_time:.2f}秒\n")
            f.write(f"原始模型平均回答长度: {base_avg_len:.1f}字符\n")
            f.write(f"微调模型平均回答长度: {tuned_avg_len:.1f}字符\n\n")
            
            # 简单评价
            if tuned_avg_len > base_avg_len * 1.2:
                f.write("✅ 微调模型回答更详细，可能学到了更多经济学知识\n")
            elif tuned_avg_len < base_avg_len * 0.8:
                f.write("⚠️ 微调模型回答较短，可能需要调整训练参数\n")
            else:
                f.write("🔍 需要人工评估回答质量来判断微调效果\n")
        
        # 打印总结
        print("\n🎉 对比评估完成!")
        print("=" * 40)
        print(f"📊 测试问题: {len(base_results)} 个")
        print(f"📦 原始模型平均用时: {base_avg_time:.2f}秒")
        print(f"🎯 微调模型平均用时: {tuned_avg_time:.2f}秒")
        print(f"📦 原始模型平均长度: {base_avg_len:.1f}字符")
        print(f"🎯 微调模型平均长度: {tuned_avg_len:.1f}字符")
        
        print(f"\n💾 详细报告已保存到: model_comparison_sequential.txt")
        print(f"📄 JSON数据已保存到: sequential_comparison.json")
        
        # 显示两个具体例子
        print(f"\n🔍 对比示例:")
        print("=" * 40)
        for i in range(min(2, len(base_results))):
            print(f"\n❓ {base_results[i]['question']}")
            print(f"📦 原始: {base_results[i]['answer'][:100]}...")
            print(f"🎯 微调: {tuned_results[i]['answer'][:100]}...")

def main():
    print("🎯 逐个模型对比评估")
    print("解决显存不足问题，逐个测试模型")
    print("=" * 50)
    
    comparator = SequentialModelComparator()
    
    try:
        base_results, tuned_results = comparator.compare_models()
        print(f"\n🎉 评估完成！共测试了 {len(base_results)} 个问题")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 