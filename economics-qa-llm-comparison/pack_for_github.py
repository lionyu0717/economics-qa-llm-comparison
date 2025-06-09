#!/usr/bin/env python3
"""
GitHub上传包生成器
"""

import os
import zipfile
from pathlib import Path

def create_package():
    print("📦 创建GitHub上传包...")
    
    # 当前目录就是项目目录
    project_dir = Path.cwd()
    
    # 输出文件到上级目录
    output_file = project_dir.parent / "economics-qa-llm-comparison-upload.zip"
    
    file_count = 0
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # 跳过这个脚本本身
            if 'pack_for_github.py' in str(root):
                continue
                
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.jsonl', '.sh', '.example')):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(project_dir)
                    zipf.write(file_path, str(rel_path))
                    file_count += 1
                    print(f"  ✅ {rel_path}")
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    
    print(f"\n🎉 完成！文件: {output_file}")
    print(f"📊 {file_count} 个文件, {file_size:.2f} MB")
    print(f"\n直接拖拽这个zip文件到GitHub就行了！")

if __name__ == "__main__":
    create_package() 