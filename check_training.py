#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练监控脚本
"""

import os
import time
import subprocess
from pathlib import Path

def check_training_status():
    """检查训练状态"""
    print("🤖 Qwen3经济学模型训练状态检查")
    print("=" * 50)
    
    # 检查GPU状态
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"🎮 GPU状态:")
            print(f"   使用率: {gpu_info[0]}%")
            print(f"   显存: {gpu_info[1]}/{gpu_info[2]} MB ({int(gpu_info[1])/int(gpu_info[2])*100:.1f}%)")
            print(f"   温度: {gpu_info[3]}°C")
        else:
            print("❌ 无法获取GPU信息")
    except:
        print("❌ nvidia-smi 不可用")
    
    print()
    
    # 检查输出目录
    output_dir = Path("fine_tuning/qwen3_economics_model")
    if output_dir.exists():
        print(f"📁 训练输出目录: {output_dir}")
        
        # 检查checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            print(f"💾 找到 {len(checkpoints)} 个检查点:")
            for cp in sorted(checkpoints):
                print(f"   - {cp.name}")
        else:
            print("💾 暂无检查点")
        
        # 检查其他文件
        files = []
        for f in output_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / 1024**2
                print(f"📄 {f.name}: {size_mb:.1f}MB")
        
        # 检查训练报告
        report_file = output_dir / "training_report.json"
        if report_file.exists():
            print("📋 找到训练报告 - 训练可能已完成!")
        
    else:
        print("📁 训练输出目录不存在，训练可能尚未开始")
    
    print()
    
    # 检查Python进程
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            python_processes = len(lines) - 1  # 减去标题行
            if python_processes > 0:
                print(f"🐍 发现 {python_processes} 个Python进程")
            else:
                print("🐍 没有发现Python进程")
        else:
            print("❌ 无法检查进程状态")
    except:
        print("❌ 无法检查进程状态")

if __name__ == "__main__":
    check_training_status() 