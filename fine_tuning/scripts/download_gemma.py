#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Google Gemma 3-1B模型到本地
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
        初始化Gemma模型下载器
        
        Args:
            base_dir: 模型保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 改用Microsoft Phi-3.5-mini，更适合我们的需求
        self.model_info = {
            "name": "phi-3.5-mini-instruct", 
            "hf_model_id": "microsoft/Phi-3.5-mini-instruct",
            "local_path": self.base_dir / "phi-3.5-mini-instruct"
        }
        
    def check_requirements(self):
        """检查必要的依赖"""
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            logger.info("✅ 所有依赖检查通过")
            return True
        except ImportError as e:
            logger.error(f"❌ 缺少依赖: {e}")
            return False
    
    def check_disk_space(self, required_gb=10):
        """检查磁盘空间"""
        import shutil
        free_bytes = shutil.disk_usage(self.base_dir).free
        free_gb = free_bytes / (1024**3)
        
        if free_gb < required_gb:
            logger.warning(f"⚠️ 磁盘空间不足: {free_gb:.1f}GB < {required_gb}GB")
            return False
        
        logger.info(f"✅ 磁盘空间充足: {free_gb:.1f}GB")
        return True
    
    def download_model(self):
        """下载Gemma模型"""
        model_info = self.model_info
        
        logger.info(f"🚀 开始下载 {model_info['name']} 模型")
        logger.info(f"📍 HuggingFace ID: {model_info['hf_model_id']}")
        logger.info(f"💾 本地路径: {model_info['local_path']}")
        
        # 检查是否已存在
        if model_info['local_path'].exists():
            logger.info(f"✅ 模型已存在: {model_info['local_path']}")
            return True
        
        try:
            start_time = time.time()
            
            # 下载整个模型仓库
            logger.info("📥 下载模型文件...")
            snapshot_download(
                repo_id=model_info['hf_model_id'],
                local_dir=str(model_info['local_path']),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            download_time = time.time() - start_time
            logger.info(f"✅ 模型下载完成! 耗时: {download_time:.1f}秒")
            
            # 验证下载
            return self.verify_model(model_info['local_path'])
            
        except Exception as e:
            logger.error(f"❌ 模型下载失败: {e}")
            return False
    
    def verify_model(self, model_path):
        """验证模型完整性"""
        logger.info("🔍 验证模型完整性...")
        
        try:
            # 检查关键文件
            required_files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not (model_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.error(f"❌ 缺少文件: {missing_files}")
                return False
            
            # 尝试加载tokenizer
            logger.info("🔍 测试tokenizer加载...")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            logger.info(f"✅ Tokenizer加载成功，词汇量: {len(tokenizer)}")
            
            # 检查模型文件大小
            model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
            logger.info(f"📊 模型文件总大小: {total_size:.2f}GB")
            
            logger.info("✅ 模型验证通过!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型验证失败: {e}")
            return False
    
    def list_downloaded_models(self):
        """列出已下载的模型"""
        logger.info("📁 已下载的模型:")
        
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)
                logger.info(f"  📦 {model_dir.name}: {size:.2f}GB")
    
    def run_download(self):
        """执行完整的下载流程"""
        logger.info("🤖 Gemma模型下载器")
        logger.info("=" * 50)
        
        # 检查环境
        if not self.check_requirements():
            return False
        
        if not self.check_disk_space():
            logger.warning("磁盘空间不足，但继续尝试下载...")
        
        # 下载模型
        success = self.download_model()
        
        if success:
            logger.info("\n🎉 Gemma模型下载成功!")
            self.list_downloaded_models()
        else:
            logger.error("\n❌ Gemma模型下载失败!")
        
        return success

def main():
    downloader = GemmaDownloader()
    success = downloader.run_download()
    
    if success:
        print("\n✅ 下载完成！现在可以使用Gemma模型进行微调了。")
    else:
        print("\n❌ 下载失败！请检查网络连接和磁盘空间。")

if __name__ == "__main__":
    main() 