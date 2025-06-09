#!/bin/bash
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
