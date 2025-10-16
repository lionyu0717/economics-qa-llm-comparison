#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控Qwen3模型训练进度
"""

import os
import time
import json
from pathlib import Path
import psutil
import subprocess

def check_training_process():
    """检查训练进程是否还在运行"""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(process.info['cmdline']) if process.info['cmdline'] else ''
            if 'train_qwen3_local.py' in cmdline:
                return process.info['pid'], process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None, None

def get_gpu_info():
    """获取GPU使用信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_info[0]),
                'memory_used': int(gpu_info[1]),
                'memory_total': int(gpu_info[2]),
                'temperature': int(gpu_info[3])
            }
    except:
        pass
    return None

def check_output_directory():
    """检查输出目录的文件"""
    output_dir = Path("fine_tuning/qwen3_economics_model")
    if not output_dir.exists():
        return "训练尚未开始或输出目录不存在"
    
    files = []
    for file in output_dir.rglob('*'):
        if file.is_file():
            files.append({
                'name': file.name,
                'path': str(file.relative_to(output_dir)),
                'size': file.stat().st_size,
                'modified': file.stat().st_mtime
            })
    
    return files

def read_training_logs():
    """读取训练日志"""
    log_patterns = [
        "fine_tuning/qwen3_economics_model/runs/",
        "fine_tuning/qwen3_economics_model/*.log",
        "fine_tuning/qwen3_economics_model/training_report.json"
    ]
    
    logs = {}
    
    # 检查训练报告
    report_file = Path("fine_tuning/qwen3_economics_model/training_report.json")
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                logs['training_report'] = json.load(f)
        except:
            pass
    
    # 检查checkpoint目录
    checkpoint_dirs = list(Path("fine_tuning/qwen3_economics_model").glob("checkpoint-*"))
    if checkpoint_dirs:
        logs['checkpoints'] = [d.name for d in sorted(checkpoint_dirs)]
    
    return logs

def monitor_training():
    """主监控函数"""
    print("🤖 Qwen3经济学模型训练监控")
    print("=" * 60)
    
    start_time = time.time()
    check_count = 0
    
    while True:
        check_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        
        print(f"\n📊 检查 #{check_count} (运行时间: {elapsed/60:.1f}分钟)")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 40)
        
        # 检查进程
        pid, process = check_training_process()
        if pid:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                print(f"✅ 训练进程运行中 (PID: {pid})")
                print(f"   CPU使用率: {cpu_percent:.1f}%")
                print(f"   内存使用: {memory_info.rss / 1024**2:.1f} MB")
            except:
                print(f"✅ 训练进程运行中 (PID: {pid})")
        else:
            print("❌ 训练进程未找到")
        
        # GPU信息
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"🎮 GPU状态:")
            print(f"   使用率: {gpu_info['gpu_util']}%")
            print(f"   显存: {gpu_info['memory_used']}/{gpu_info['memory_total']} MB ({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
            print(f"   温度: {gpu_info['temperature']}°C")
        
        # 检查输出文件
        files = check_output_directory()
        if isinstance(files, list):
            print(f"📁 输出文件 ({len(files)}个):")
            recent_files = sorted(files, key=lambda x: x['modified'], reverse=True)[:5]
            for f in recent_files:
                size_mb = f['size'] / 1024**2
                mod_time = time.strftime('%H:%M:%S', time.localtime(f['modified']))
                print(f"   {f['name']}: {size_mb:.1f}MB (修改时间: {mod_time})")
        else:
            print(f"📁 {files}")
        
        # 检查训练日志
        logs = read_training_logs()
        if logs.get('checkpoints'):
            print(f"💾 检查点: {', '.join(logs['checkpoints'])}")
        
        if logs.get('training_report'):
            report = logs['training_report']
            print(f"📋 训练报告:")
            print(f"   最终训练损失: {report.get('final_train_loss', 'N/A')}")
            print(f"   最终验证损失: {report.get('final_eval_loss', 'N/A')}")
            print(f"   测试样本数: {report.get('test_samples', 'N/A')}")
        
        # 如果进程不存在且有训练报告，说明训练完成
        if not pid and logs.get('training_report'):
            print("\n🎉 训练已完成!")
            break
        
        # 如果进程不存在且运行时间超过5分钟，可能是异常退出
        if not pid and elapsed > 300:
            print("\n⚠️ 训练进程不存在，可能已异常退出")
            response = input("是否继续监控? (y/N): ")
            if response.lower() != 'y':
                break
        
        # 等待30秒后下次检查
        print("\n⏳ 等待30秒后继续监控...")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n⏹️ 监控被用户中断")
            break

if __name__ == "__main__":
    monitor_training() 