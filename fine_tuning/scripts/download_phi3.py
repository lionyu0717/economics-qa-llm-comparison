#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Google Gemma-3-1B模型到本地
用于与Qwen3模型进行对比微调
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GemmaDownloader:
    def __init__(self, base_dir="fine_tuning/models"):
        """
        初始化Gemma-3-1B模型下载器
        
        Args:
            base_dir: 模型保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用Google Gemma 3-1B，更适合对比
        self.model_info = {
            "name": "gemma-3-1b-it", 
            "hf_model_id": "google/gemma-3-1b-it",
            "local_path": self.base_dir / "gemma-3-1b-it"
        }
        
        # 配置HuggingFace token
        self.hf_token = "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"
    
    def check_dependencies(self):
        """检查必要的依赖"""
        try:
            import torch
            import transformers
            import huggingface_hub
            logger.info("✅ 所有依赖检查通过")
            return True
        except ImportError as e:
            logger.error(f"❌ 缺少依赖: {e}")
            return False
    
    def check_disk_space(self, required_gb=10):
        """检查磁盘空间"""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.base_dir.parent).free
            free_gb = free_bytes / (1024**3)
            
            if free_gb >= required_gb:
                logger.info(f"✅ 磁盘空间充足: {free_gb:.1f}GB")
                return True
            else:
                logger.error(f"❌ 磁盘空间不足: {free_gb:.1f}GB (需要 {required_gb}GB)")
                return False
        except Exception as e:
            logger.warning(f"⚠️ 无法检查磁盘空间: {e}")
            return True
    
    def download_model(self):
        """下载模型"""
        model_info = self.model_info
        
        logger.info("🚀 开始下载 {} 模型".format(model_info["name"]))
        logger.info("📍 HuggingFace ID: {}".format(model_info["hf_model_id"]))
        
        try:
            if model_info["local_path"].exists():
                logger.info(f"✅ 模型已存在: {model_info['local_path']}")
                return True
            
            logger.info("📥 下载模型文件...")
            snapshot_download(
                repo_id=model_info["hf_model_id"],
                local_dir=str(model_info["local_path"]),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=self.hf_token  # 使用提供的token
            )
            
            logger.info("✅ {} 模型下载成功!".format(model_info["name"]))
            logger.info("📁 保存位置: {}".format(model_info["local_path"]))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型下载失败: {e}")
            return False
    
    def verify_model(self):
        """验证下载的模型"""
        model_info = self.model_info
        
        try:
            logger.info("🔍 验证模型文件...")
            
            # 检查关键文件
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            missing_files = []
            
            for file in required_files:
                if not (model_info["local_path"] / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.error(f"❌ 缺少关键文件: {missing_files}")
                return False
            
            # 尝试加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_info["local_path"]))
            logger.info("✅ Tokenizer加载成功")
            
            logger.info("✅ 模型验证通过!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型验证失败: {e}")
            return False
    
    def run(self):
        """运行下载流程"""
        logger.info("🤖 Gemma-3-1B模型下载器")
        logger.info("=" * 50)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 检查磁盘空间
        if not self.check_disk_space():
            return False
        
        # 下载模型
        if not self.download_model():
            logger.error("❌ 下载失败！请检查网络连接和磁盘空间。")
            return False
        
        # 验证模型
        if not self.verify_model():
            logger.error("❌ 模型验证失败！")
            return False
        
        logger.info("")
        logger.info("🎉 Gemma-3-1B模型下载完成！")
        logger.info(f"📁 模型位置: {self.model_info['local_path']}")
        logger.info("✨ 现在可以开始微调了!")
        
        return True

if __name__ == "__main__":
    downloader = GemmaDownloader()
    success = downloader.run()
    
    if not success:
        sys.exit(1) 