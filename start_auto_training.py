#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动启动训练脚本
自动开始QWEN和LLAMA模型的微调训练
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def start_qwen_training():
    """启动QWEN训练"""
    print("="*50)
    print("🚀 启动QWEN模型微调")
    print("="*50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("模型: Qwen2-1.5B-Instruct")
    print("数据: 100个经济学问答对（快速训练）")
    print("方法: LoRA微调")
    print()
    
    try:
        result = subprocess.run([
            sys.executable, "fine_tuning/scripts/train_qwen_simple.py"
        ], capture_output=False, text=True, cwd=".")
        
        return result.returncode == 0
    except Exception as e:
        print(f"QWEN训练失败: {e}")
        return False

def start_llama_training():
    """启动LLAMA训练"""
    print("="*50)
    print("🚀 启动LLAMA模型微调")
    print("="*50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("模型: DialoGPT-medium (作为LLAMA替代)")
    print("数据: 100个经济学问答对（快速训练）")
    print("方法: LoRA微调")
    print()
    
    try:
        result = subprocess.run([
            sys.executable, "fine_tuning/scripts/train_llama_simple.py"
        ], capture_output=False, text=True, cwd=".")
        
        return result.returncode == 0
    except Exception as e:
        print(f"LLAMA训练失败: {e}")
        return False

def check_environment():
    """检查环境"""
    print("🔍 检查训练环境...")
    
    # 检查GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"GPU可用: {'✅' if gpu_available else '❌'}")
        if gpu_available:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查数据
    data_files = [
        "fine_tuning/data/qwen/train.jsonl",
        "fine_tuning/data/alpaca/train.json"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ 数据文件存在: {file_path}")
        else:
            print(f"❌ 数据文件缺失: {file_path}")
            return False
    
    return True

def monitor_progress():
    """监控训练进度"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("📊 训练进度监控")
        print("="*40)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查QWEN进度
        qwen_checkpoint = "fine_tuning/qwen/checkpoints"
        if os.path.exists(qwen_checkpoint):
            qwen_files = [f for f in os.listdir(qwen_checkpoint) if f.endswith('.bin') or f.endswith('.safetensors')]
            print(f"QWEN: ✅ {len(qwen_files)} 个检查点文件")
        else:
            print("QWEN: ⏳ 训练中...")
        
        # 检查LLAMA进度
        llama_checkpoint = "fine_tuning/llama/checkpoints"
        if os.path.exists(llama_checkpoint):
            llama_files = [f for f in os.listdir(llama_checkpoint) if f.endswith('.bin') or f.endswith('.safetensors')]
            print(f"LLAMA: ✅ {len(llama_files)} 个检查点文件")
        else:
            print("LLAMA: ⏳ 训练中...")
        
        print("\n按 Ctrl+C 停止监控")
        time.sleep(10)

def main():
    print("🎯 经济学问答助手模型微调启动器")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请检查依赖和数据")
        return
    
    print("✅ 环境检查通过")
    print()
    
    # 启动QWEN训练
    print("1️⃣ 开始QWEN模型微调...")
    qwen_success = start_qwen_training()
    
    if qwen_success:
        print("✅ QWEN训练完成")
    else:
        print("❌ QWEN训练失败")
    
    print("\n" + "="*50)
    print("等待5秒后开始LLAMA训练...")
    time.sleep(5)
    
    # 启动LLAMA训练
    print("2️⃣ 开始LLAMA模型微调...")
    llama_success = start_llama_training()
    
    if llama_success:
        print("✅ LLAMA训练完成")
    else:
        print("❌ LLAMA训练失败")
    
    # 总结
    print("\n" + "="*60)
    print("🏁 训练总结")
    print("="*60)
    print(f"QWEN模型: {'✅ 成功' if qwen_success else '❌ 失败'}")
    print(f"LLAMA模型: {'✅ 成功' if llama_success else '❌ 失败'}")
    
    if qwen_success or llama_success:
        print("\n🎉 至少有一个模型训练成功！")
        print("接下来可以运行模型测试:")
        print("python fine_tuning/scripts/test_simple.py")
    else:
        print("\n😞 所有模型训练都失败了，请检查错误信息")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断训练")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 