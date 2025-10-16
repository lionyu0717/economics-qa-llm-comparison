#!/usr/bin/env python3
"""
简单的GitHub上传包生成器
创建一个干净的项目压缩包，便于手动上传到GitHub
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_upload_package():
    """创建GitHub上传包"""
    print("📦 创建GitHub上传包...")
    
    # 当前目录
    current_dir = Path.cwd()
    
    # 输出文件名
    output_file = current_dir.parent / "economics-qa-llm-comparison-github.zip"
    
    # 要包含的文件和目录
    include_patterns = [
        "README.md",
        "LICENSE", 
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "src/",
        "docs/",
        "configs/",
        "scripts/",
        "data/",
        "results/",
        "tests/"
    ]
    
    # 排除的文件模式
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    def should_include(file_path):
        """判断文件是否应该包含"""
        file_str = str(file_path)
        
        # 检查排除模式
        for pattern in exclude_patterns:
            if pattern in file_str:
                return False
        
        return True
    
    # 创建压缩包
    file_count = 0
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            path = current_dir / pattern
            
            if path.is_file():
                if should_include(path):
                    zipf.write(path, path.name)
                    file_count += 1
                    print(f"  ✅ {path.name}")
                    
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    # 过滤目录
                    dirs[:] = [d for d in dirs if should_include(Path(root) / d)]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if should_include(file_path):
                            # 计算相对路径
                            rel_path = file_path.relative_to(current_dir)
                            zipf.write(file_path, str(rel_path))
                            file_count += 1
                            print(f"  ✅ {rel_path}")
    
    # 获取文件大小
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    
    print("\n" + "="*50)
    print("🎉 上传包创建完成！")
    print("="*50)
    print(f"📁 文件位置: {output_file}")
    print(f"📊 包含文件: {file_count} 个")
    print(f"💾 文件大小: {file_size:.2f} MB")
    print("\n📋 使用方法:")
    print("1. 登录 https://github.com")
    print("2. 点击右上角 '+' → 'New repository'")
    print("3. 仓库名: economics-qa-llm-comparison")
    print("4. 选择 'Public' 并创建仓库")
    print("5. 在仓库页面点击 'uploading an existing file'")
    print("6. 拖拽或选择刚才创建的zip文件上传")
    print("7. 写个提交信息，比如: 'Initial commit: Economics QA project'")
    print("8. 点击 'Commit changes'")
    print("\n🎯 就这么简单！")

if __name__ == "__main__":
    create_upload_package() 