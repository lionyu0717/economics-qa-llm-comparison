#!/bin/bash
# 模型训练脚本

echo "🚀 开始训练模型..."

# 训练Qwen3
echo "📊 训练Qwen3模型..."
python src/training/train_qwen3.py

# 训练Gemma3
echo "📊 训练Gemma3模型..."
python src/training/train_gemma3.py

echo "✅ 训练完成!"
