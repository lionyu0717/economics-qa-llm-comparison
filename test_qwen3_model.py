#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试已训练完成的Qwen3经济学模型
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_trained_model():
    """加载训练好的模型"""
    print("🤖 加载Qwen3经济学问答模型...")
    
    base_model_path = "fine_tuning/models/qwen3-1.7b"
    trained_model_path = "fine_tuning/qwen3_economics_model"
    
    # 检查路径
    if not Path(base_model_path).exists():
        print(f"❌ 基础模型路径不存在: {base_model_path}")
        return None, None
    
    if not Path(trained_model_path).exists():
        print(f"❌ 训练模型路径不存在: {trained_model_path}")
        return None, None
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, trained_model_path)
        model.eval()
        
        print("✅ 模型加载成功!")
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def test_economics_questions(model, tokenizer):
    """测试经济学问题"""
    
    # 准备测试问题
    test_questions = [
        "什么是稀缺性？",
        "请解释供给与需求的关系",
        "什么是机会成本？",
        "请说明价格弹性的概念",
        "什么是市场失灵？",
        "解释通货膨胀的含义",
        "什么是GDP？",
        "请说明垄断市场的特点"
    ]
    
    print("📝 开始测试经济学问答...")
    print("=" * 60)
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n❓ 问题 {i+1}: {question}")
        print("-" * 40)
        
        # 构建输入
        input_text = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        try:
            # 生成回答
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
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.replace(input_text, "").strip()
            
            print(f"💡 回答: {answer}")
            
            results.append({
                "question": question,
                "answer": answer
            })
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            results.append({
                "question": question,
                "answer": f"生成失败: {e}"
            })
    
    return results

def save_test_results(results):
    """保存测试结果"""
    output_file = "qwen3_economics_test_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

def main():
    """主函数"""
    print("🎯 Qwen3经济学问答模型测试")
    print("=" * 50)
    
    # 加载模型
    model, tokenizer = load_trained_model()
    if model is None:
        return
    
    # 测试问题
    results = test_economics_questions(model, tokenizer)
    
    # 保存结果
    save_test_results(results)
    
    print("\n🎉 测试完成!")
    print(f"总共测试了 {len(results)} 个问题")
    
    # 显示GPU状态
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU显存使用: {memory_used:.1f}/{memory_total:.1f} GB")

if __name__ == "__main__":
    main() 