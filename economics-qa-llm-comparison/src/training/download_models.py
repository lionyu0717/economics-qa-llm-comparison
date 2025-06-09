#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
下载最新的小参数模型到本地进行离线微调
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)
from huggingface_hub import snapshot_download
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, base_dir="fine_tuning/models"):
        """
        初始化模型下载器
        
        Args:
            base_dir: 模型保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义要下载的模型
        self.models_to_download = {
            "qwen3-1.7b": {
                "model_name": "Qwen/Qwen3-1.7B", 
                "local_dir": self.base_dir / "qwen3-1.7b",
                "description": "Qwen3 1.7B 基础模型"
            },
            "qwen3-1.7b-instruct": {
                "model_name": "Qwen/Qwen3-1.7B-Instruct",
                "local_dir": self.base_dir / "qwen3-1.7b-instruct", 
                "description": "Qwen3 1.7B 指令微调模型"
            },
            "llama3.2-3b": {
                "model_name": "meta-llama/Llama-3.2-3B",
                "local_dir": self.base_dir / "llama3.2-3b",
                "description": "Llama 3.2 3B 基础模型"
            },
            "llama3.2-3b-instruct": {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct", 
                "local_dir": self.base_dir / "llama3.2-3b-instruct",
                "description": "Llama 3.2 3B 指令微调模型"
            }
        }
        
    def check_disk_space(self, required_gb=20):
        """
        检查磁盘空间
        
        Args:
            required_gb: 需要的最小磁盘空间(GB)
        """
        import shutil
        free_bytes = shutil.disk_usage(self.base_dir).free
        free_gb = free_bytes / (1024**3)
        
        if free_gb < required_gb:
            logger.warning(f"磁盘剩余空间只有 {free_gb:.1f}GB，建议至少有 {required_gb}GB")
            return False
        else:
            logger.info(f"磁盘剩余空间: {free_gb:.1f}GB ✅")
            return True
    
    def download_model_files(self, model_name, local_dir, description=""):
        """
        下载模型文件到本地目录
        
        Args:
            model_name: HuggingFace模型名称
            local_dir: 本地保存目录
            description: 模型描述
        """
        logger.info(f"开始下载 {description}: {model_name}")
        logger.info(f"保存位置: {local_dir}")
        
        try:
            # 创建目录
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已经下载过
            if (local_dir / "config.json").exists():
                logger.info(f"模型 {model_name} 已存在，跳过下载")
                return True
            
            start_time = time.time()
            
            # 使用 snapshot_download 下载整个模型仓库
            logger.info("正在下载模型文件...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
                resume_download=True,  # 支持断点续传
                ignore_patterns=["*.msgpack", "*.h5", "original/*"]  # 忽略不需要的文件
            )
            
            # 验证下载的文件
            essential_files = ["config.json", "tokenizer.json"]
            for file in essential_files:
                if not (local_dir / file).exists():
                    logger.warning(f"关键文件 {file} 不存在")
            
            end_time = time.time()
            logger.info(f"✅ {description} 下载完成! 耗时: {end_time - start_time:.1f}秒")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载 {model_name} 失败: {str(e)}")
            return False
    
    def test_model_loading(self, model_name, local_dir, description=""):
        """
        测试模型是否能正确加载
        
        Args:
            model_name: 模型名称
            local_dir: 本地目录
            description: 模型描述
        """
        logger.info(f"测试加载 {description}...")
        
        try:
            # 测试加载配置
            config = AutoConfig.from_pretrained(str(local_dir))
            logger.info(f"配置加载成功: {config.model_type}")
            
            # 测试加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
            logger.info(f"Tokenizer加载成功: 词汇量 {tokenizer.vocab_size}")
            
            # 测试编码解码
            test_text = "你好，这是一个测试。"
            tokens = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(tokens)
            logger.info(f"编码解码测试成功: {len(tokens)} tokens")
            
            # 检查GPU内存
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU内存: {gpu_memory:.1f}GB")
                
                # 估算是否能加载完整模型
                if "1.7b" in model_name.lower() or "3b" in model_name.lower():
                    logger.info("小模型，GPU内存应该足够加载完整模型")
                else:
                    logger.info("大模型，建议使用量化或者模型并行")
            
            logger.info(f"✅ {description} 加载测试成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ {description} 加载测试失败: {str(e)}")
            return False
    
    def download_all_models(self):
        """
        下载所有模型
        """
        logger.info("🚀 开始下载模型到本地...")
        logger.info(f"保存位置: {self.base_dir.absolute()}")
        
        # 检查磁盘空间
        if not self.check_disk_space(30):  # 需要30GB空间
            response = input("磁盘空间可能不足，是否继续? (y/N): ")
            if response.lower() != 'y':
                logger.info("下载已取消")
                return
        
        success_count = 0
        total_count = len(self.models_to_download)
        
        for model_key, model_info in self.models_to_download.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"下载进度: {success_count + 1}/{total_count}")
            
            success = self.download_model_files(
                model_info["model_name"],
                model_info["local_dir"],
                model_info["description"]
            )
            
            if success:
                # 测试模型加载
                test_success = self.test_model_loading(
                    model_info["model_name"],
                    model_info["local_dir"],
                    model_info["description"]
                )
                if test_success:
                    success_count += 1
                    logger.info(f"✅ {model_info['description']} 完全就绪!")
                else:
                    logger.warning(f"⚠️ {model_info['description']} 下载成功但加载测试失败")
            else:
                logger.error(f"❌ {model_info['description']} 下载失败")
            
            # 短暂暂停，避免请求过于频繁
            time.sleep(2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 下载完成!")
        logger.info(f"成功: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("🎉 所有模型都已成功下载并验证!")
            self.show_model_info()
        else:
            logger.warning(f"⚠️ 有 {total_count - success_count} 个模型下载失败")
    
    def show_model_info(self):
        """
        显示已下载模型的信息
        """
        logger.info("\n📋 已下载的模型信息:")
        logger.info("-" * 80)
        
        for model_key, model_info in self.models_to_download.items():
            local_dir = model_info["local_dir"]
            if local_dir.exists():
                try:
                    config_file = local_dir / "config.json"
                    if config_file.exists():
                        import json
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        size_mb = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file()) / (1024*1024)
                        
                        logger.info(f"📁 {model_info['description']}")
                        logger.info(f"   路径: {local_dir}")
                        logger.info(f"   模型类型: {config.get('model_type', 'unknown')}")
                        logger.info(f"   参数数量: {config.get('num_parameters', 'unknown')}")
                        logger.info(f"   文件大小: {size_mb:.1f} MB")
                        logger.info(f"   词汇表大小: {config.get('vocab_size', 'unknown')}")
                        logger.info("")
                except Exception as e:
                    logger.info(f"📁 {model_info['description']}: {local_dir} (无法读取详细信息)")

def main():
    """
    主函数
    """
    print("🤖 模型下载工具")
    print("准备下载 Qwen3-1.7B 和 Llama-3.2-3B 模型...")
    print()
    
    downloader = ModelDownloader()
    
    try:
        downloader.download_all_models()
    except KeyboardInterrupt:
        logger.info("\n⏹️ 下载被用户中断")
    except Exception as e:
        logger.error(f"❌ 下载过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 