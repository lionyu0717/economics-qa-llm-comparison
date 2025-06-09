#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的模型评估脚本
快速测试微调效果
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class SimpleEvaluator:
    def __init__(self):
        self.base_model_path = "fine_tuning/models/qwen3-1.7b"
        self.tuned_model_path = "fine_tuning/qwen3_economics_model"
        self.data_path = "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
    
    def load_test_questions(self):
        """加载测试问题"""
        test_questions = [
            "什么是稀缺性？",
            "请解释供给与需求的关系",
            "什么是机会成本？",
            "请说明价格弹性的概念",
            "什么是市场失灵？",
            "解释通货膨胀的含义",
            "什么是比较优势？",
            "请说明生产可能性边界的概念"
        ]
        
        # 从数据文件中获取标准答案
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # 匹配问题和答案
        qa_pairs = []
        for question in test_questions:
            for item in data:
                if question in item["question"] or item["question"] in question:
                    qa_pairs.append({
                        "question": question,
                        "reference": item["answer"]
                    })
                    break
            else:
                qa_pairs.append({
                    "question": question,
                    "reference": "标准答案未找到"
                })
        
        return qa_pairs
    
    def load_models(self):
        """加载模型"""
        print("🤖 加载模型...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载微调模型
        print("🎯 加载微调模型...")
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
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.replace(input_text, "").strip()
            
            # 清理答案
            if "\n" in answer:
                answer = answer.split("\n")[0]
            
            return answer[:200]
            
        except Exception as e:
            return f"生成失败: {str(e)}"
    
    def simple_similarity(self, ref, cand):
        """简单的相似度计算"""
        ref_words = set(ref.split())
        cand_words = set(cand.split())
        
        if len(ref_words) == 0 and len(cand_words) == 0:
            return 1.0
        
        intersection = len(ref_words & cand_words)
        union = len(ref_words | cand_words)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate(self):
        """运行评估"""
        print("🎯 简化版微调效果评估")
        print("=" * 40)
        
        # 加载测试数据
        qa_pairs = self.load_test_questions()
        print(f"📚 加载了 {len(qa_pairs)} 个测试问题")
        
        # 加载模型
        tokenizer, tuned_model = self.load_models()
        
        print("\n📊 开始测试...")
        print("=" * 60)
        
        results = []
        total_similarity = 0
        
        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            reference = qa["reference"]
            
            print(f"\n❓ 问题 {i+1}: {question}")
            print("-" * 40)
            
            # 生成回答
            generated = self.generate_answer(tuned_model, tokenizer, question)
            
            # 计算相似度
            similarity = self.simple_similarity(reference, generated)
            total_similarity += similarity
            
            print(f"🎯 标准答案: {reference[:100]}...")
            print(f"💡 模型回答: {generated}")
            print(f"📈 相似度: {similarity:.3f}")
            
            results.append({
                "question": question,
                "reference": reference,
                "generated": generated,
                "similarity": similarity
            })
        
        # 总结
        avg_similarity = total_similarity / len(qa_pairs)
        print(f"\n🎉 评估完成!")
        print("=" * 40)
        print(f"📊 平均相似度: {avg_similarity:.3f}")
        
        if avg_similarity > 0.3:
            print("✅ 微调效果良好！模型能够较好地回答经济学问题。")
        elif avg_similarity > 0.15:
            print("⚠️ 微调效果一般，有一定改善但还需优化。")
        else:
            print("❌ 微调效果不明显，需要调整训练策略。")
        
        # 保存结果
        with open("simple_evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果已保存到: simple_evaluation_results.json")
        
        return results

def main():
    evaluator = SimpleEvaluator()
    try:
        results = evaluator.evaluate()
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 