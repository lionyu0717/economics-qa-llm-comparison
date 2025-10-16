# 经济学问答助手：Qwen3 vs Gemma3 微调对比研究

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.51+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 基于《经济学原理》教材的大语言模型微调与对比评估项目

## 🎯 项目概述

本项目构建了一个经济学问答助手，通过对比Qwen3-1.7B和Gemma3-1B两个先进的大语言模型，深入研究了模型微调、参数优化和性能评估的完整流程。

### 核心亮点
- 📚 **高质量数据集**: 1026个基于权威教材的经济学问答对
- 🤖 **双模型对比**: Qwen3-1.7B vs Gemma3-1B全面性能评估
- ⚡ **技术突破**: 解决Qwen3无限重复等关键技术难题
- 📊 **科学评估**: 多维度评估体系，包含速度、质量、稳定性指标
- 🛠️ **完整流程**: 从数据处理到模型部署的完整实现

## 📈 主要发现

| 指标 | Qwen3-1.7B | Gemma3-1B | 优势方 |
|------|------------|-----------|--------|
| **响应速度** | 23.78秒 | 3.59秒 | Gemma3 **快6.6倍** |
| **回答长度** | 249字符 | 92字符 | Qwen3 **详细2.7倍** |
| **稳定性** | 有重复(已修复) | 稳定 | Gemma3 |
| **教学价值** | 详细完整 | 简洁精准 | 各有优势 |

## 🔧 技术特色

### 解决的关键问题
1. **Qwen3无限重复**: 发现并解决了贪婪解码导致的重复问题
2. **显存优化**: 8bit量化 + LoRA实现单GPU双模型训练
3. **数据适配**: 统一不同模型的数据格式和评估标准
4. **参数调优**: 基于官方建议优化生成参数

### 技术栈
- **深度学习**: PyTorch, Transformers, PEFT
- **模型架构**: LoRA微调, 8bit量化
- **评估工具**: 自定义多维度评估框架
- **数据处理**: 经济学文本预处理和问答生成

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.8
CUDA >= 11.8 (推荐)
GPU显存 >= 8GB
```

### 安装依赖
```bash
git clone https://github.com/yourusername/economics-qa-llm-comparison.git
cd economics-qa-llm-comparison
pip install -r requirements.txt
```

### 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件，添加必要的API密钥
```

### 运行演示
```bash
# 数据处理
python src/data_processing/generate_qa.py

# 模型训练
python src/training/train_qwen3.py
python src/training/train_gemma3.py

# 模型对比评估
python src/evaluation/model_comparison.py
```

## 📁 项目结构

```
├── src/                    # 核心源代码
│   ├── data_processing/    # 数据处理模块
│   ├── training/          # 模型训练模块
│   ├── evaluation/        # 模型评估模块
│   └── utils/             # 工具函数
├── docs/                  # 项目文档
├── configs/               # 配置文件
├── scripts/               # 运行脚本
├── data/                  # 数据目录
└── results/               # 结果输出
```

## 📊 实验结果

### 性能对比
- **Qwen3-1.7B**: 适合需要详细解释的教学场景
- **Gemma3-1B**: 适合快速响应的实时问答应用

### 训练效果
- **数据规模**: 1026条专业问答对
- **训练效率**: 6.064 samples/second
- **损失改善**: 从4.11降至2.11 (48.7%提升)

### 技术突破
解决了Qwen3模型在微调后的无限重复问题，这是该模型在实际应用中的重要技术突破。

## 🎓 应用场景

### 教育领域
- **在线课程**: 24小时经济学答疑服务
- **学习辅助**: 个性化经济学概念解释
- **考试准备**: 经济学知识点快速检索

### 研究领域
- **学术研究**: 经济学概念标准化解释
- **知识管理**: 经济学知识库构建
- **模型研究**: 大语言模型微调技术验证

## 🔬 技术细节

### 微调策略
- **LoRA配置**: rank=16, alpha=32, dropout=0.1
- **训练参数**: 3 epochs, lr=5e-5, batch_size=4
- **优化技术**: 8bit量化, 梯度累积

### 评估方法
- **多维指标**: 响应时间、回答质量、稳定性
- **对比分析**: 同一测试集下的公平评估
- **量化评估**: 字符长度、生成速度等客观指标

## 📄 引用

如果此项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{economics-qa-llm-2025,
  title={Economics Q&A Assistant: A Comparative Study of Qwen3 vs Gemma3 Fine-tuning},
  author={YourName},
  year={2025},
  url={https://github.com/yourusername/economics-qa-llm-comparison}
}
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请确保：
- 代码符合项目规范
- 添加必要的测试
- 更新相关文档

## 📧 联系方式

- 项目维护者: [YourName](mailto:your.email@example.com)
- 项目地址: https://github.com/yourusername/economics-qa-llm-comparison

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！ 