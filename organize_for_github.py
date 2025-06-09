#!/usr/bin/env python3
"""
GitHub项目整理脚本
自动化整理项目文件，准备上传到GitHub
"""

import os
import shutil
import json
from pathlib import Path
import yaml

class GitHubOrganizer:
    def __init__(self):
        self.current_dir = Path.cwd()
        self.github_dir = self.current_dir / "economics-qa-llm-comparison"
        
    def create_directory_structure(self):
        """创建标准的GitHub项目目录结构"""
        print("🏗️ 创建项目目录结构...")
        
        directories = [
            "src/data_processing",
            "src/training", 
            "src/evaluation",
            "src/utils",
            "docs",
            "configs",
            "scripts",
            "data/raw",
            "data/processed",
            "results/evaluation_reports",
            "results/figures",
            "tests"
        ]
        
        for directory in directories:
            (self.github_dir / directory).mkdir(parents=True, exist_ok=True)
            # 创建__init__.py文件
            if directory.startswith("src/"):
                (self.github_dir / directory / "__init__.py").touch()
                
        print("✅ 目录结构创建完成")
    
    def copy_core_files(self):
        """复制核心文件到新的目录结构"""
        print("📁 复制核心文件...")
        
        file_mappings = {
            # 文档
            "项目总结报告.md": "docs/项目总结报告.md",
            "PPT展示大纲.md": "docs/PPT展示大纲.md",
            "GitHub-README.md": "README.md",
            
            # 数据处理
            "extract_text.py": "src/data_processing/extract_text.py",
            "preprocess_text.py": "src/data_processing/preprocess_text.py", 
            "generate_qa.py": "src/data_processing/generate_qa.py",
            "generate_1000_qa.py": "src/data_processing/generate_1000_qa.py",
            
            # 模型训练
            "fine_tuning/scripts/train_qwen3_local.py": "src/training/train_qwen3.py",
            "fine_tuning/scripts/train_gemma3_direct.py": "src/training/train_gemma3.py",
            "fine_tuning/scripts/download_models.py": "src/training/download_models.py",
            
            # 模型评估
            "model_comparison_evaluation.py": "src/evaluation/model_comparison.py",
            "comprehensive_evaluation.py": "src/evaluation/comprehensive_evaluation.py",
            "complete_model_evaluation.py": "src/evaluation/complete_evaluation.py",
            
            # 脚本
            "test_qwen3_fixed_parameters.py": "scripts/test_qwen3_parameters.py",
            "run_pipeline.py": "scripts/run_pipeline.py",
            
            # 配置
            "requirements.txt": "requirements.txt",
            "env.example": ".env.example",
            ".gitignore": ".gitignore"
        }
        
        for source, target in file_mappings.items():
            source_path = self.current_dir / source
            target_path = self.github_dir / target
            
            if source_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                print(f"  ✅ {source} → {target}")
            else:
                print(f"  ⚠️ 文件不存在: {source}")
    
    def create_sample_data(self):
        """创建示例数据文件"""
        print("📊 创建示例数据...")
        
        # 从原始数据集中提取前10条作为示例
        original_data_path = self.current_dir / "经济学原理 (N.格里高利曼昆) (Z-Library)_dataset_openrouter/qa_dataset_1000/qa_pairs_1000.jsonl"
        sample_data_path = self.github_dir / "data/raw/sample_data.jsonl"
        
        if original_data_path.exists():
            with open(original_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # 只取前10条
            
            with open(sample_data_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("  ✅ 示例数据创建完成")
        else:
            # 创建一个简单的示例
            sample_data = [
                {
                    "question": "什么是生产要素？",
                    "answer": "生产要素是指用于生产物品和劳务的任何东西，包括土地、劳动力、资本和企业家才能。"
                },
                {
                    "question": "什么是边际产量？",
                    "answer": "边际产量是指在保持其他投入不变时，增加一单位某种投入所引起的产量变化。"
                }
            ]
            
            with open(sample_data_path, 'w', encoding='utf-8') as f:
                for item in sample_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print("  ✅ 默认示例数据创建完成")
    
    def create_config_files(self):
        """创建配置文件"""
        print("⚙️ 创建配置文件...")
        
        # Qwen3配置
        qwen3_config = {
            "model_name": "Qwen/Qwen3-1.7B",
            "training": {
                "learning_rate": 5e-5,
                "epochs": 3,
                "batch_size": 4,
                "max_length": 1024
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            "generation": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.2,
                "max_new_tokens": 512
            }
        }
        
        # Gemma3配置
        gemma3_config = {
            "model_name": "google/gemma-3-1b-it",
            "training": {
                "learning_rate": 5e-5,
                "epochs": 3,
                "batch_size": 4,
                "max_length": 1024
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            "quantization": {
                "load_in_8bit": True,
                "device_map": "auto"
            }
        }
        
        # 评估配置
        evaluation_config = {
            "test_samples": 30,
            "metrics": ["response_time", "answer_length", "quality_score"],
            "comparison_models": ["qwen3", "gemma3"],
            "output_format": "json"
        }
        
        with open(self.github_dir / "configs/qwen3_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(qwen3_config, f, default_flow_style=False, allow_unicode=True)
        
        with open(self.github_dir / "configs/gemma3_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(gemma3_config, f, default_flow_style=False, allow_unicode=True)
        
        with open(self.github_dir / "configs/evaluation_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(evaluation_config, f, default_flow_style=False, allow_unicode=True)
        
        print("  ✅ 配置文件创建完成")
    
    def create_shell_scripts(self):
        """创建Shell运行脚本"""
        print("🚀 创建运行脚本...")
        
        # 训练脚本
        train_script = """#!/bin/bash
# 模型训练脚本

echo "🚀 开始训练模型..."

# 训练Qwen3
echo "📊 训练Qwen3模型..."
python src/training/train_qwen3.py

# 训练Gemma3
echo "📊 训练Gemma3模型..."
python src/training/train_gemma3.py

echo "✅ 训练完成!"
"""
        
        # 评估脚本
        eval_script = """#!/bin/bash
# 模型评估脚本

echo "🔍 开始模型评估..."

# 运行对比评估
python src/evaluation/model_comparison.py

# 运行综合评估
python src/evaluation/comprehensive_evaluation.py

echo "✅ 评估完成!"
"""
        
        # 设置脚本
        setup_script = """#!/bin/bash
# 环境设置脚本

echo "🛠️ 设置项目环境..."

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 复制环境配置
cp .env.example .env

echo "✅ 环境设置完成!"
echo "请编辑 .env 文件，添加必要的API密钥"
"""
        
        scripts = {
            "scripts/train_models.sh": train_script,
            "scripts/evaluate_models.sh": eval_script,
            "scripts/setup.sh": setup_script
        }
        
        for script_path, content in scripts.items():
            with open(self.github_dir / script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            # 添加执行权限
            os.chmod(self.github_dir / script_path, 0o755)
        
        print("  ✅ 运行脚本创建完成")
    
    def create_data_readme(self):
        """创建数据目录说明"""
        data_readme = """# 数据目录说明

## 📁 目录结构

```
data/
├── raw/                    # 原始数据
│   └── sample_data.jsonl   # 示例数据(10条)
└── processed/              # 处理后数据
    └── (训练时生成)
```

## 📊 数据格式

### 原始数据格式 (JSONL)
```json
{
  "question": "什么是生产要素？",
  "answer": "生产要素是指用于生产物品和劳务的任何东西，包括土地、劳动力、资本和企业家才能。"
}
```

## 🔄 数据处理流程

1. **文本提取**: 从教材中提取经济学内容
2. **预处理**: 清理和格式化文本
3. **问答生成**: 使用AI生成问答对
4. **质量控制**: 人工审核和修正

## 📝 使用说明

- `sample_data.jsonl`: 包含10条示例数据，用于测试
- 完整数据集包含1026条专业经济学问答对
- 数据基于《经济学原理》(N.格里高利·曼昆)教材生成

## ⚠️ 注意事项

- 完整数据集因版权原因未上传
- 可使用示例数据进行功能测试
- 如需完整数据集，请参考项目文档重新生成
"""
        
        with open(self.github_dir / "data/README.md", 'w', encoding='utf-8') as f:
            f.write(data_readme)
        print("  ✅ 数据说明文档创建完成")
    
    def copy_results(self):
        """复制重要的结果文件"""
        print("📊 复制结果文件...")
        
        result_files = [
            "model_comparison_report_20250609_160928.txt",
            "comprehensive_evaluation_report_20250609_173854.txt"
        ]
        
        for file_name in result_files:
            source_path = self.current_dir / file_name
            if source_path.exists():
                target_path = self.github_dir / "results/evaluation_reports" / file_name
                shutil.copy2(source_path, target_path)
                print(f"  ✅ {file_name}")
    
    def create_license(self):
        """创建MIT许可证"""
        mit_license = """MIT License

Copyright (c) 2025 Economics QA LLM Comparison Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        with open(self.github_dir / "LICENSE", 'w', encoding='utf-8') as f:
            f.write(mit_license)
        print("  ✅ LICENSE文件创建完成")
    
    def run_organization(self):
        """运行完整的整理流程"""
        print("🎯 开始整理GitHub项目...")
        print("=" * 50)
        
        # 删除已存在的目录
        if self.github_dir.exists():
            shutil.rmtree(self.github_dir)
        
        # 创建目录结构
        self.create_directory_structure()
        
        # 复制核心文件
        self.copy_core_files()
        
        # 创建示例数据
        self.create_sample_data()
        
        # 创建配置文件
        self.create_config_files()
        
        # 创建运行脚本
        self.create_shell_scripts()
        
        # 创建数据说明
        self.create_data_readme()
        
        # 复制结果文件
        self.copy_results()
        
        # 创建许可证
        self.create_license()
        
        print("=" * 50)
        print("🎉 项目整理完成!")
        print(f"📁 项目位置: {self.github_dir}")
        print("\n📋 下一步操作:")
        print("1. cd economics-qa-llm-comparison")
        print("2. git init")
        print("3. git add .")
        print("4. git commit -m 'Initial commit: Economics QA LLM Comparison Project'")
        print("5. 在GitHub创建仓库并推送")
        print("\n🔧 记得编辑以下文件:")
        print("- README.md (更新个人信息和GitHub链接)")
        print("- .env.example (添加必要的API配置)")

if __name__ == "__main__":
    organizer = GitHubOrganizer()
    organizer.run_organization() 