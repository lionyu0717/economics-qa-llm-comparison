#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
带监控功能的训练脚本
可以监控训练进度并保存日志
"""

import subprocess
import sys
import os
import time
import threading
from datetime import datetime

class TrainingMonitor:
    def __init__(self, log_dir="fine_tuning/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def run_training(self, script_name, model_name):
        """运行训练并监控"""
        print(f"开始 {model_name} 模型微调...")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建日志文件
        log_file = os.path.join(self.log_dir, f"{model_name.lower()}_training.log")
        
        try:
            # 启动训练进程
            with open(log_file, 'w', encoding='utf-8') as log:
                process = subprocess.Popen([
                    sys.executable, script_name
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, cwd=".", bufsize=1, universal_newlines=True)
                
                print(f"训练进程已启动，PID: {process.pid}")
                print(f"日志文件: {log_file}")
                print("实时输出:")
                print("-" * 50)
                
                # 实时显示输出
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line.rstrip())
                        log.write(line)
                        log.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    print(f"\n{model_name} 训练成功完成!")
                    return True
                else:
                    print(f"\n{model_name} 训练失败，返回码: {process.returncode}")
                    return False
                    
        except Exception as e:
            print(f"{model_name} 训练过程中出错: {e}")
            return False

def check_training_progress():
    """检查训练进度"""
    print("检查训练进度...")
    
    qwen_checkpoint = "fine_tuning/qwen/checkpoints"
    llama_checkpoint = "fine_tuning/llama/checkpoints"
    
    # 检查QWEN进度
    if os.path.exists(qwen_checkpoint):
        qwen_files = os.listdir(qwen_checkpoint)
        print(f"QWEN检查点: {len(qwen_files)} 个文件")
        for f in qwen_files[:5]:  # 显示前5个文件
            print(f"  - {f}")
    else:
        print("QWEN: 尚未开始训练")
    
    # 检查LLAMA进度
    if os.path.exists(llama_checkpoint):
        llama_files = os.listdir(llama_checkpoint)
        print(f"LLAMA检查点: {len(llama_files)} 个文件")
        for f in llama_files[:5]:  # 显示前5个文件
            print(f"  - {f}")
    else:
        print("LLAMA: 尚未开始训练")

def main():
    print("="*60)
    print("经济学问答助手模型微调监控器")
    print("="*60)
    
    monitor = TrainingMonitor()
    
    # 检查当前进度
    check_training_progress()
    
    print("\n选择操作:")
    print("1. 启动QWEN训练")
    print("2. 启动LLAMA训练") 
    print("3. 启动两个模型训练")
    print("4. 检查进度")
    print("5. 测试已有模型")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == "1":
        success = monitor.run_training("fine_tuning/scripts/train_qwen_simple.py", "QWEN")
        if success:
            print("\n是否继续训练LLAMA模型? (y/n): ", end="")
            if input().lower() == 'y':
                monitor.run_training("fine_tuning/scripts/train_llama_simple.py", "LLAMA")
    
    elif choice == "2":
        monitor.run_training("fine_tuning/scripts/train_llama_simple.py", "LLAMA")
    
    elif choice == "3":
        print("依次训练两个模型...")
        qwen_success = monitor.run_training("fine_tuning/scripts/train_qwen_simple.py", "QWEN")
        if qwen_success:
            print("\n等待5秒后开始LLAMA训练...")
            time.sleep(5)
            monitor.run_training("fine_tuning/scripts/train_llama_simple.py", "LLAMA")
        else:
            print("QWEN训练失败，跳过LLAMA训练")
    
    elif choice == "4":
        check_training_progress()
        
        # 检查日志文件
        log_dir = "fine_tuning/logs"
        if os.path.exists(log_dir):
            log_files = os.listdir(log_dir)
            if log_files:
                print(f"\n可用日志文件:")
                for f in log_files:
                    print(f"  - {f}")
                
                log_choice = input("\n输入日志文件名查看最后20行 (回车跳过): ").strip()
                if log_choice and log_choice in log_files:
                    log_path = os.path.join(log_dir, log_choice)
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            print(f"\n{log_choice} 最后20行:")
                            print("-" * 50)
                            for line in lines[-20:]:
                                print(line.rstrip())
                    except Exception as e:
                        print(f"读取日志失败: {e}")
    
    elif choice == "5":
        print("运行模型测试...")
        try:
            subprocess.run([sys.executable, "fine_tuning/scripts/test_simple.py"], cwd=".")
        except Exception as e:
            print(f"测试失败: {e}")
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main() 