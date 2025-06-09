#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Qwen3模型 - 使用官方推荐的最佳参数
解决重复回答问题
"""

import torch
import time
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qwen3_model():
    """加载Qwen3微调模型"""
    try:
        print("🤖 加载Qwen3模型...")
        
        # 模型路径
        base_model_path = "fine_tuning/models/qwen3-1.7b"
        adapter_path = "fine_tuning/qwen3_economics_model"
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  # 使用原始数据类型
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # 使用8bit量化节省显存
        )
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        print("✅ 模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def generate_answer_optimized(model, tokenizer, question, mode="non_thinking"):
    """
    使用优化参数生成回答
    mode: "thinking" 或 "non_thinking"
    """
    try:
        # 构建prompt - 使用Qwen3标准格式
        if mode == "thinking":
            # 使用thinking模式的标准格式
            messages = [{"role": "user", "content": f"你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。\n{question}"}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
        else:
            # 使用简单格式，非thinking模式
            prompt = f"<|user|>\n你是一个专业的经济学问答助手，请根据经济学原理准确回答问题。\n{question}\n<|assistant|>\n"
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 根据模式设置不同的生成参数
        if mode == "thinking":
            # Thinking模式参数（官方推荐）
            generation_params = {
                "max_new_tokens": 200,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.2,  # 增加重复惩罚
            }
        else:
            # Non-thinking模式参数（官方推荐）
            generation_params = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.3,  # 更高的重复惩罚
            }
        
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_params
            )
        
        generation_time = time.time() - start_time
        
        # 解码输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if mode == "thinking":
            # 处理thinking模式的输出
            if "<think>" in full_response and "</think>" in full_response:
                # 有thinking过程
                think_start = full_response.find("<think>")
                think_end = full_response.find("</think>") + 8
                thinking_content = full_response[think_start:think_end]
                answer = full_response[think_end:].strip()
                print(f"💭 推理过程: {thinking_content[:100]}...")
            else:
                answer = full_response.replace(prompt, "").strip()
        else:
            answer = full_response.replace(prompt, "").strip()
        
        # 清理答案
        if "\n" in answer:
            lines = answer.split("\n")
            # 取第一个非空行作为主要答案
            for line in lines:
                if line.strip():
                    answer = line.strip()
                    break
        
        # 限制答案长度
        if len(answer) > 300:
            answer = answer[:300] + "..."
        
        return answer, generation_time
        
    except Exception as e:
        return f"生成失败: {str(e)}", 0

def test_economics_questions():
    """测试经济学问题"""
    
    # 加载模型
    model, tokenizer = load_qwen3_model()
    if model is None:
        return
    
    # 测试问题
    test_questions = [
        "什么是稀缺性？",
        "请解释供给与需求的关系",
        "什么是机会成本？",
        "什么是边际效用？",
        "请说明生产要素的概念"
    ]
    
    print("\n🧪 开始测试Qwen3优化参数...")
    print("=" * 70)
    
    results = []
    
    for mode in ["non_thinking", "thinking"]:
        print(f"\n📋 测试模式: {mode.upper()}")
        print("-" * 50)
        
        mode_results = []
        
        for i, question in enumerate(test_questions):
            print(f"\n❓ 问题 {i+1}: {question}")
            
            answer, gen_time = generate_answer_optimized(model, tokenizer, question, mode)
            
            print(f"⏱️ 生成时间: {gen_time:.2f}秒")
            print(f"💡 回答: {answer}")
            
            mode_results.append({
                "question": question,
                "answer": answer,
                "generation_time": gen_time,
                "answer_length": len(answer)
            })
            
            print("-" * 30)
        
        results.append({
            "mode": mode,
            "results": mode_results,
            "avg_time": sum(r["generation_time"] for r in mode_results) / len(mode_results),
            "avg_length": sum(r["answer_length"] for r in mode_results) / len(mode_results)
        })
    
    # 输出对比报告
    print("\n📊 参数优化效果对比")
    print("=" * 70)
    
    for result in results:
        mode = result["mode"]
        avg_time = result["avg_time"]
        avg_length = result["avg_length"]
        
        print(f"📋 {mode.upper()} 模式:")
        print(f"   ⏱️ 平均生成时间: {avg_time:.2f}秒")
        print(f"   📏 平均回答长度: {avg_length:.1f}字符")
        
        # 检查重复问题
        repetition_count = 0
        for r in result["results"]:
            answer = r["answer"]
            # 简单检查重复（相同句子出现多次）
            sentences = answer.split("。")
            if len(sentences) > 2:
                unique_sentences = set(sentences)
                if len(unique_sentences) < len(sentences) * 0.8:  # 80%以上句子重复
                    repetition_count += 1
        
        print(f"   🔄 重复回答数量: {repetition_count}/{len(result['results'])}")
        print()
    
    # 保存结果
    report_file = f"qwen3_parameter_optimization_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"📄 详细报告已保存: {report_file}")
    
    return results

def main():
    """主函数"""
    print("🎯 Qwen3参数优化测试")
    print("解决重复回答问题，对比thinking vs non-thinking模式")
    print("=" * 70)
    
    try:
        results = test_economics_questions()
        print("\n✅ 测试完成！")
        
        if results:
            print("\n💡 优化建议:")
            print("1. 使用官方推荐的参数组合")
            print("2. 增加repetition_penalty到1.2-1.3")
            print("3. 根据应用场景选择thinking或non-thinking模式")
            print("4. 考虑使用presence_penalty参数（如果框架支持）")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 