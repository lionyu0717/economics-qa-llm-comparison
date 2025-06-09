#!/bin/bash
# 模型评估脚本

echo "🔍 开始模型评估..."

# 运行对比评估
python src/evaluation/model_comparison.py

# 运行综合评估
python src/evaluation/comprehensive_evaluation.py

echo "✅ 评估完成!"
