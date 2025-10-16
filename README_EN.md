# Economics Q&A Assistant - LLM Comparison and Fine-tuning Project

[![GitHub](https://img.shields.io/badge/GitHub-economics--qa--llm--comparison-blue)](https://github.com/lionyu0717/economics-qa-llm-comparison)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI application focused on economics domain question-answering systems. Based on the textbook "Principles of Economics" by N. Gregory Mankiw, this project implements a complete workflow from text extraction, dataset generation, model fine-tuning to performance evaluation, comparing multiple large language models on economics Q&A tasks.

## 📚 Key Features

- **Complete Data Processing Pipeline**: EPUB text extraction → Intelligent chunking → Automatic Q&A generation
- **OpenRouter Integration**: Support for multiple LLMs via OpenRouter (Gemini, Mixtral, Claude, etc.)
- **Model Fine-tuning**: Domain-specific fine-tuning support for models like Qwen
- **Performance Evaluation**: Comprehensive model comparison and evaluation system
- **RAG Integration**: Integration with RAGflow for retrieval-augmented generation
- **High-Quality Dataset**: Automatically generated economics Q&A dataset

## 🏗️ Project Architecture

```
.
├── 📁 Data Processing Module
│   ├── extract_text.py          # EPUB text extraction
│   ├── preprocess_text.py       # Text preprocessing and chunking
│   ├── generate_qa.py           # Q&A pair generation (OpenRouter)
│   └── run_pipeline.py          # Complete pipeline script
│
├── 📁 Model Training & Evaluation
│   ├── fine_tuning/             # Model fine-tuning scripts
│   ├── compare_models.py        # Model performance comparison
│   ├── comprehensive_evaluation.py  # Comprehensive evaluation system
│   └── simple_evaluation.py     # Simple evaluation tool
│
├── 📁 Datasets
│   ├── economics_dataset_openrouter/  # Generated datasets
│   │   ├── chunks/              # Text chunks
│   │   ├── qa_dataset/          # Q&A pairs
│   │   │   ├── train.jsonl      # Training set
│   │   │   ├── val.jsonl        # Validation set
│   │   │   └── test.jsonl       # Test set
│   │   └── processing_summary_openrouter.json
│   └── example/                 # Example data
│
├── 📁 Configuration Files
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example             # Environment variable template
│   └── README.md                # Project documentation
│
└── 📁 Utility Scripts
    ├── test_openrouter_api.py   # API connection test
    └── create_upload_package.py # Data packaging tool
```

## 🚀 Quick Start

### 1. Environment Setup

#### Create Virtual Environment (Recommended)

```bash
# Using Conda
conda create -n economics_qa python=3.9
conda activate economics_qa

# Or using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure API Keys

Copy `.env.example` to `.env` and configure your API keys:

```env
# OpenRouter API Key (Required)
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Optional Configuration
YOUR_SITE_URL=https://your-site-url.com
YOUR_SITE_NAME=Economics QA Project
```

> 📝 **Get API Key**: Visit [OpenRouter.ai](https://openrouter.ai/keys) to register and obtain

### 2. Test API Connection

```bash
python test_openrouter_api.py
```

Upon success, you'll see the model's response.

### 3. Generate Q&A Dataset

#### Method 1: One-Click Processing (Recommended)

```bash
python run_pipeline.py "principles_of_economics.epub"
```

#### Method 2: Step-by-Step Execution

```bash
# Step 1: Extract text
python extract_text.py "principles_of_economics.epub" --save-chapters

# Step 2: Chunk text
python preprocess_text.py "./economics_chapters" --chunk-size 500 --overlap 50

# Step 3: Generate Q&A pairs
python generate_qa.py "./chunks" --model "google/gemini-2.0-flash-001" --questions 3
```

### 4. Advanced Options

#### Custom Data Generation

```bash
python run_pipeline.py "principles_of_economics.epub" \
    --output custom_dataset \
    --chunk-size 400 \
    --questions 5 \
    --sample 0.5 \
    --model "anthropic/claude-3-sonnet"
```

#### Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output`, `-o` | Output directory name | `{book}_dataset_openrouter` |
| `--chunk-size` | Text chunk size (words) | 500 |
| `--overlap` | Text chunk overlap (words) | 50 |
| `--questions`, `-q` | Questions per chunk | 3 |
| `--sample`, `-s` | Text processing ratio (0.0-1.0) | 1.0 |
| `--model`, `-m` | LLM model to use | `google/gemini-2.0-flash-001` |
| `--no-chapters` | Don't save separate chapter files | False |

## 📊 Dataset Description

### Dataset Formats

#### 1. Basic Q&A Format (`qa_pairs_openrouter.jsonl`)

```json
{
  "question": "What is opportunity cost?",
  "answer": "Opportunity cost is whatever must be given up to obtain something...",
  "chapter": "Chapter One: Ten Principles of Economics",
  "chunk_id": "chunk_002_001"
}
```

#### 2. Prompt-Response Format (`prompt_response_openrouter.jsonl`)

```json
{
  "prompt": "Please answer the following economics question: What is opportunity cost?",
  "response": "Opportunity cost is whatever must be given up to obtain something..."
}
```

### Dataset Split

- **Training Set (train.jsonl)**: 80%
- **Validation Set (val.jsonl)**: 10%
- **Test Set (test.jsonl)**: 10%

## 🤖 Supported Models

### Available Models via OpenRouter

- **Google Gemini Series**: `google/gemini-2.0-flash-001`, `google/gemini-pro`
- **Anthropic Claude Series**: `anthropic/claude-3-sonnet`, `anthropic/claude-3-opus`
- **Meta Llama Series**: `meta-llama/llama-3-70b-instruct`
- **Mistral Series**: `mistralai/mixtral-8x7b-instruct`
- **More Models**: Check [OpenRouter Documentation](https://openrouter.ai/docs)

### Recommended Model Configurations

| Use Case | Recommended Model | Advantages |
|----------|------------------|------------|
| Fast Generation | `google/gemini-2.0-flash-001` | Fast, low cost |
| High Quality | `anthropic/claude-3-sonnet` | High quality, strong reasoning |
| Balanced | `mistralai/mixtral-8x7b-instruct` | Cost-effective |

## 📈 Model Evaluation & Comparison

The project includes a complete model evaluation system:

```bash
# Run model comparison evaluation
python compare_models.py

# Run comprehensive evaluation
python comprehensive_evaluation.py
```

Evaluation metrics include:
- **Accuracy**: Correctness of answers
- **Completeness**: Detail level of answers
- **Relevance**: Relevance to questions
- **Fluency**: Naturalness of language

## 🔧 Model Fine-tuning

### Prepare Training Data

The dataset is automatically split into training, validation, and test sets in formats compatible with mainstream fine-tuning frameworks.

### Fine-tuning Example (Using Qwen)

```bash
# See scripts in fine_tuning/ directory
cd fine_tuning
python train_qwen.py
```

## 📝 Output File Description

Complete output directory structure after processing:

```
economics_dataset_openrouter/
├── raw_text/                    # Raw extracted text
│   └── economics_extracted.txt
├── chapters/                    # Text split by chapters
│   ├── 001_principles.txt
│   ├── 002_chapter_one.txt
│   └── ...
├── chunks/                      # Text chunks (JSON format)
│   ├── chunk_001_001.json
│   ├── chunks_index.json        # Chunk index file
│   └── ...
└── qa_dataset/                  # Q&A dataset
    ├── qa_pairs_openrouter.jsonl          # Original Q&A pairs
    ├── prompt_response_openrouter.jsonl   # Prompt-Response format
    ├── split_openrouter/                  # Dataset split
    │   ├── train.jsonl          # Training set (80%)
    │   ├── val.jsonl            # Validation set (10%)
    │   └── test.jsonl           # Test set (10%)
    └── processing_summary_openrouter.json # Processing summary
```

## 💡 Best Practices

### 1. Data Quality Optimization

- **Adjust chunk_size**: Smaller chunks (300-400 words) for specific questions, larger chunks (600-800 words) for comprehensive questions
- **Control overlap**: Appropriate overlap (50-100 words) ensures context continuity
- **Choose appropriate model**: Gemini for fast iteration, Claude for high-quality data

### 2. Cost Control

- **Use sample parameter**: Test with `--sample 0.1` first
- **Choose cost-effective models**: Flash models have low cost, suitable for large-scale generation
- **Monitor API usage**: Check usage in OpenRouter console

### 3. Improve Q&A Quality

- **Adjust prompts**: Modify prompt templates in `generate_qa.py`
- **Increase questions**: Generate more questions per chunk (3-5)
- **Manual review**: Sample check generated Q&A pair quality

## 🐛 Troubleshooting

### Common Issues

#### 1. API Connection Failure

```bash
# Test connection
python test_openrouter_api.py

# Check:
# - Is API key correct?
# - Is network connection normal?
# - Does OpenRouter account have sufficient balance?
```

#### 2. EPUB Parsing Error

```bash
# Solutions:
# - Convert EPUB format using Calibre
# - Check if EPUB file is corrupted
# - Try other EPUB files
```

#### 3. JSON Parsing Error

```bash
# Script includes automatic retry mechanism
# If persistent failure, may be model output format issue
# Suggestion: Switch to more reliable model (like Claude)
```

#### 4. Out of Memory

```bash
# Reduce chunk_size
python run_pipeline.py input.epub --chunk-size 300

# Or process in batches
python run_pipeline.py input.epub --sample 0.3
```

## 📖 Usage Examples

### Example 1: Quick Small-Scale Dataset Generation

```bash
python run_pipeline.py "principles_of_economics.epub" --sample 0.2 --questions 3
```

### Example 2: Generate High-Quality Complete Dataset

```bash
python run_pipeline.py "principles_of_economics.epub" \
    --model "anthropic/claude-3-sonnet" \
    --questions 5 \
    --chunk-size 600
```

### Example 3: Quick Test Different Models

```bash
# Using Gemini (fast)
python generate_qa.py ./chunks --model "google/gemini-2.0-flash-001"

# Using Claude (high quality)
python generate_qa.py ./chunks --model "anthropic/claude-3-sonnet"
```

## 🔐 Important Notes

1. **API Key Security**: 
   - Don't commit `.env` file to Git
   - Use `.gitignore` to exclude sensitive files

2. **Cost Control**:
   - Different models have different pricing, check OpenRouter pricing
   - Use `--sample` parameter for small-scale testing first

3. **Copyright Compliance**:
   - This project is for academic research and personal learning only
   - Please comply with textbook copyright and API terms of use

4. **Data Quality**:
   - Recommend manual sampling of generated Q&A pairs
   - May require post-processing and cleaning

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork this repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details

## 📮 Contact

- **GitHub Issues**: [Submit Issue](https://github.com/lionyu0717/economics-qa-llm-comparison/issues)
- **Project Homepage**: [economics-qa-llm-comparison](https://github.com/lionyu0717/economics-qa-llm-comparison)

## 🙏 Acknowledgments

- N. Gregory Mankiw, author of "Principles of Economics"
- OpenRouter for API services
- Contributors to all open-source dependencies

## 📊 Project Status

- ✅ Data extraction and preprocessing
- ✅ Q&A pair generation
- ✅ Dataset splitting
- ✅ Model evaluation framework
- 🚧 Model fine-tuning (in progress)
- 🚧 RAGflow integration (in progress)

---

**⭐ If this project helps you, please give us a Star!**

*This project is for academic research and learning purposes only. Please comply with relevant laws, regulations, and API terms of use.*

