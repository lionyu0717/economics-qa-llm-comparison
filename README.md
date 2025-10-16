# 经济学问答助手 - LLM模型对比与微调项目

[![GitHub](https://img.shields.io/badge/GitHub-economics--qa--llm--comparison-blue)](https://github.com/lionyu0717/economics-qa-llm-comparison)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本项目是一个完整的人工智能应用，专注于经济学领域的问答系统。项目基于《经济学原理》(曼昆)教材，实现了从文本提取、数据集生成、模型微调到性能评估的完整流程，并对比了多个大语言模型在经济学问答任务上的表现。

## 📚 项目特色

- **完整的数据处理流程**：从EPUB文件提取文本 → 智能分块 → 自动生成问答对
- **OpenRouter集成**：支持通过OpenRouter调用多种大语言模型（Gemini、Mixtral、Claude等）
- **模型微调**：支持对Qwen等模型进行领域特定微调
- **性能评估**：全面的模型对比与评估系统
- **RAG集成**：集成RAGflow实现检索增强生成
- **高质量数据集**：自动生成的经济学问答数据集

## 🏗️ 项目架构

```
.
├── 📁 数据处理模块
│   ├── extract_text.py          # EPUB文本提取
│   ├── preprocess_text.py       # 文本预处理和分块
│   ├── generate_qa.py           # 问答对生成（OpenRouter）
│   └── run_pipeline.py          # 完整流水线脚本
│
├── 📁 模型训练与评估
│   ├── fine_tuning/             # 模型微调脚本
│   ├── compare_models.py        # 模型性能对比
│   ├── comprehensive_evaluation.py  # 综合评估系统
│   └── simple_evaluation.py     # 简易评估工具
│
├── 📁 数据集
│   ├── 经济学原理_dataset_openrouter/  # 生成的数据集
│   │   ├── chunks/              # 文本分块
│   │   ├── qa_dataset/          # 问答对数据
│   │   │   ├── train.jsonl      # 训练集
│   │   │   ├── val.jsonl        # 验证集
│   │   │   └── test.jsonl       # 测试集
│   │   └── processing_summary_openrouter.json
│   └── example/                 # 示例数据
│
├── 📁 配置文件
│   ├── requirements.txt         # Python依赖
│   ├── .env.example             # 环境变量模板
│   └── README.md                # 项目文档
│
└── 📁 工具脚本
    ├── test_openrouter_api.py   # API连接测试
    └── create_upload_package.py # 数据打包工具
```

## 🚀 快速开始

### 1. 环境配置

#### 创建虚拟环境（推荐）

```bash
# 使用 Conda
conda create -n economics_qa python=3.9
conda activate economics_qa

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 配置API密钥

复制 `.env.example` 为 `.env` 并配置您的API密钥：

```env
# OpenRouter API密钥（必需）
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# 可选配置
YOUR_SITE_URL=https://your-site-url.com
YOUR_SITE_NAME=Economics QA Project
```

> 📝 **获取API密钥**：访问 [OpenRouter.ai](https://openrouter.ai/keys) 注册并获取

### 2. 测试API连接

```bash
python test_openrouter_api.py
```

成功后会显示模型的响应内容。

### 3. 生成问答数据集

#### 方式一：一键式处理（推荐）

```bash
python run_pipeline.py "经济学原理.epub"
```

#### 方式二：分步执行

```bash
# 步骤1: 提取文本
python extract_text.py "经济学原理.epub" --save-chapters

# 步骤2: 文本分块
python preprocess_text.py "./经济学原理_chapters" --chunk-size 500 --overlap 50

# 步骤3: 生成问答对
python generate_qa.py "./chunks" --model "google/gemini-2.0-flash-001" --questions 3
```

### 4. 高级选项

#### 自定义数据生成

```bash
python run_pipeline.py "经济学原理.epub" \
    --output custom_dataset \
    --chunk-size 400 \
    --questions 5 \
    --sample 0.5 \
    --model "anthropic/claude-3-sonnet"
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output`, `-o` | 输出目录名称 | `{书名}_dataset_openrouter` |
| `--chunk-size` | 文本块大小（词数） | 500 |
| `--overlap` | 文本块重叠量（词数） | 50 |
| `--questions`, `-q` | 每块生成的问题数 | 3 |
| `--sample`, `-s` | 处理的文本比例（0.0-1.0） | 1.0 |
| `--model`, `-m` | 使用的LLM模型 | `google/gemini-2.0-flash-001` |
| `--no-chapters` | 不保存单独章节文件 | False |

## 📊 数据集说明

### 数据集格式

#### 1. 基础问答格式 (`qa_pairs_openrouter.jsonl`)

```json
{
  "question": "什么是机会成本？",
  "answer": "机会成本是指为了得到某种东西所必须放弃的东西...",
  "chapter": "第一章 经济学十大原理",
  "chunk_id": "chunk_002_001"
}
```

#### 2. Prompt-Response格式 (`prompt_response_openrouter.jsonl`)

```json
{
  "prompt": "请回答以下经济学问题：什么是机会成本？",
  "response": "机会成本是指为了得到某种东西所必须放弃的东西..."
}
```

### 数据集划分

- **训练集 (train.jsonl)**: 80%
- **验证集 (val.jsonl)**: 10%
- **测试集 (test.jsonl)**: 10%

## 🤖 支持的模型

### 通过OpenRouter可用的模型

- **Google Gemini系列**: `google/gemini-2.0-flash-001`, `google/gemini-pro`
- **Anthropic Claude系列**: `anthropic/claude-3-sonnet`, `anthropic/claude-3-opus`
- **Meta Llama系列**: `meta-llama/llama-3-70b-instruct`
- **Mistral系列**: `mistralai/mixtral-8x7b-instruct`
- **更多模型**: 查看 [OpenRouter文档](https://openrouter.ai/docs)

### 推荐模型配置

| 用途 | 推荐模型 | 优势 |
|------|---------|------|
| 快速生成 | `google/gemini-2.0-flash-001` | 速度快，成本低 |
| 高质量输出 | `anthropic/claude-3-sonnet` | 质量高，推理能力强 |
| 平衡选择 | `mistralai/mixtral-8x7b-instruct` | 性价比高 |

## 📈 模型评估与对比

项目包含完整的模型评估系统：

```bash
# 运行模型对比评估
python compare_models.py

# 运行综合评估
python comprehensive_evaluation.py
```

评估指标包括：
- **准确性**: 答案的正确性
- **完整性**: 答案的详细程度
- **相关性**: 与问题的相关程度
- **流畅性**: 语言表达的自然度

## 🔧 模型微调

### 准备训练数据

数据集会自动划分为训练集、验证集和测试集，格式符合主流微调框架要求。

### 微调示例（使用Qwen）

```bash
# 详见 fine_tuning/ 目录中的脚本
cd fine_tuning
python train_qwen.py
```

## 📝 输出文件说明

完整处理后的输出目录结构：

```
经济学原理_dataset_openrouter/
├── raw_text/                    # 原始提取的文本
│   └── 经济学原理_extracted.txt
├── chapters/                    # 按章节分割的文本
│   ├── 001_经济学原理.txt
│   ├── 002_第一章_经济学十大原理.txt
│   └── ...
├── chunks/                      # 文本块（JSON格式）
│   ├── chunk_001_001.json
│   ├── chunks_index.json        # 块索引文件
│   └── ...
└── qa_dataset/                  # 问答数据集
    ├── qa_pairs_openrouter.jsonl          # 原始问答对
    ├── prompt_response_openrouter.jsonl   # Prompt-Response格式
    ├── split_openrouter/                  # 数据集划分
    │   ├── train.jsonl          # 训练集（80%）
    │   ├── val.jsonl            # 验证集（10%）
    │   └── test.jsonl           # 测试集（10%）
    └── processing_summary_openrouter.json # 处理摘要
```

## 💡 最佳实践

### 1. 数据质量优化

- **调整chunk_size**: 较小的块（300-400词）适合生成具体问题，较大的块（600-800词）适合综合性问题
- **控制overlap**: 适当的重叠（50-100词）确保上下文连贯性
- **选择合适的模型**: Gemini适合快速迭代，Claude适合生成高质量数据

### 2. 成本控制

- **使用sample参数**: 先用 `--sample 0.1` 测试效果
- **选择合适的模型**: Flash模型成本低，适合大规模生成
- **监控API使用**: 在OpenRouter控制台查看使用情况

### 3. 提高问答质量

- **调整prompt**: 修改 `generate_qa.py` 中的提示词模板
- **增加questions数量**: 每个chunk生成更多问题（3-5个）
- **人工审核**: 抽样检查生成的问答对质量

## 🐛 故障排除

### 常见问题

#### 1. API连接失败

```bash
# 测试连接
python test_openrouter_api.py

# 检查事项：
# - API密钥是否正确
# - 网络连接是否正常
# - OpenRouter账户余额是否充足
```

#### 2. EPUB解析错误

```bash
# 解决方案：
# - 使用Calibre转换EPUB格式
# - 检查EPUB文件是否损坏
# - 尝试其他EPUB文件
```

#### 3. JSON解析错误

```bash
# 脚本包含自动重试机制
# 如果持续失败，可能是模型输出格式问题
# 建议：切换到更可靠的模型（如Claude）
```

#### 4. 内存不足

```bash
# 减小chunk_size
python run_pipeline.py input.epub --chunk-size 300

# 或分批处理
python run_pipeline.py input.epub --sample 0.3
```

## 📖 使用示例

### 示例1: 快速生成小规模数据集

```bash
python run_pipeline.py "经济学原理.epub" --sample 0.2 --questions 3
```

### 示例2: 生成高质量完整数据集

```bash
python run_pipeline.py "经济学原理.epub" \
    --model "anthropic/claude-3-sonnet" \
    --questions 5 \
    --chunk-size 600
```

### 示例3: 快速测试不同模型

```bash
# 使用Gemini（快速）
python generate_qa.py ./chunks --model "google/gemini-2.0-flash-001"

# 使用Claude（高质量）
python generate_qa.py ./chunks --model "anthropic/claude-3-sonnet"
```

## 🔐 注意事项

1. **API密钥安全**: 
   - 不要将 `.env` 文件提交到Git
   - 使用 `.gitignore` 排除敏感文件

2. **成本控制**:
   - 不同模型价格差异大，请查看OpenRouter定价
   - 使用 `--sample` 参数先小规模测试

3. **版权合规**:
   - 本项目仅用于学术研究和个人学习
   - 请遵守教材版权和API使用条款

4. **数据质量**:
   - 生成的问答对建议人工抽查
   - 可能需要后处理和清洗

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📮 联系方式

- **GitHub Issues**: [提交问题](https://github.com/lionyu0717/economics-qa-llm-comparison/issues)
- **项目主页**: [economics-qa-llm-comparison](https://github.com/lionyu0717/economics-qa-llm-comparison)

## 🙏 致谢

- 《经济学原理》作者 N. Gregory Mankiw
- OpenRouter提供的API服务
- 所有开源依赖库的贡献者

## 📊 项目状态

- ✅ 数据提取和预处理
- ✅ 问答对生成
- ✅ 数据集划分
- ✅ 模型评估框架
- 🚧 模型微调（进行中）
- 🚧 RAGflow集成（进行中）

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**

*本项目仅用于学术研究和学习目的，请遵守相关法律法规和API使用条款。*
