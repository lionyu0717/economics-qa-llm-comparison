#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单模型测试脚本
测试微调后的模型是否能正常工作
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_qwen_model():
    """测试QWEN模型"""
    base_model_name = "Qwen/Qwen2-1.5B-Instruct"
    checkpoint_path = "./fine_tuning/qwen/checkpoints"
    
    if not os.path.exists(checkpoint_path):
        print("QWEN检查点不存在，跳过测试")
        return
    
    try:
        print("加载QWEN模型...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # 测试问题
        question = "什么是稀缺性？"
        prompt = f"<|im_start|>system\n你是一位专业的经济学助手。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("<|im_start|>assistant\n")[-1]
        
        print(f"QWEN回答: {answer}")
        print("QWEN模型测试成功！")
        
    except Exception as e:
        print(f"QWEN模型测试失败: {e}")

def test_llama_model():
    """测试LLAMA(DialoGPT)模型"""
    base_model_name = "microsoft/DialoGPT-medium"
    checkpoint_path = "./fine_tuning/llama/checkpoints"
    
    if not os.path.exists(checkpoint_path):
        print("LLAMA检查点不存在，跳过测试")
        return
    
    try:
        print("加载LLAMA模型...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # 测试问题
        question = "什么是稀缺性？"
        prompt = f"### Instruction:\n请回答以下经济学问题：\n\n### Input:\n{question}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Response:\n")[-1]
        
        print(f"LLAMA回答: {answer}")
        print("LLAMA模型测试成功！")
        
    except Exception as e:
        print(f"LLAMA模型测试失败: {e}")

def main():
    print("="*50)
    print("模型测试")
    print("="*50)
    
    # 检查GPU状态
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print("\n" + "-"*30)
    test_qwen_model()
    
    print("\n" + "-"*30)
    test_llama_model()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 