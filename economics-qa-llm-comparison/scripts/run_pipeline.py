#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整的经济学教材问答数据集构建流水线
用法：python run_pipeline.py path/to/epub_file.epub
"""

import os
import sys
import argparse
import time
import subprocess
import json
from extract_text import epub_to_text
from preprocess_text import process_single_text_file, process_chapters_directory
import shlex
from dotenv import load_dotenv

# 加载.env文件，这会把.env中的变量设置到环境变量中
load_dotenv()

def run_command(command):
    """运行shell命令并打印输出"""
    print(f"执行命令: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def create_project_structure(base_dir):
    """创建项目目录结构"""
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建所需的子目录
    dirs = [
        "raw_text",
        "chapters",
        "chunks",
        "qa_dataset"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    print(f"已创建项目目录结构: {base_dir}")
    return {d: os.path.join(base_dir, d) for d in dirs}

def run_pipeline(epub_path, output_dir=None, save_chapters=True, chunk_size=500, overlap=50, 
                model="google/gemini-2.0-flash-001", questions_per_chunk=3, sample_ratio=1.0):
    """运行完整的数据处理流水线"""
    
    start_time = time.time()
    
    # 如果没有提供输出目录，使用EPUB文件名作为基础
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(epub_path))[0]
        output_dir = f"{base_name}_dataset_openrouter"
    
    # 创建项目结构
    dirs = create_project_structure(output_dir)
    
    # 第一步：从EPUB提取文本
    print("\n===== 步骤1：从EPUB提取文本 =====")
    raw_text_file = os.path.join(dirs['raw_text'], f"{os.path.splitext(os.path.basename(epub_path))[0]}_extracted.txt")
    chapters_dir = dirs['chapters']
    
    try:
        epub_to_text(epub_path, raw_text_file, save_chapters=save_chapters)
        if save_chapters:
            source_chapters_dir = f"{os.path.splitext(os.path.basename(epub_path))[0]}_chapters"
            if os.path.exists(source_chapters_dir) and source_chapters_dir != chapters_dir:
                import shutil
                if os.path.exists(chapters_dir):
                    shutil.rmtree(chapters_dir)
                shutil.copytree(source_chapters_dir, chapters_dir)
                print(f"已将章节文件复制到项目目录: {chapters_dir}")
    except Exception as e:
        print(f"提取文本时出错: {e}")
        sys.exit(1)
    
    # 第二步：预处理文本并分块
    print("\n===== 步骤2：预处理文本并分块 =====")
    chunks_dir = dirs['chunks']
    
    try:
        if save_chapters and os.path.exists(chapters_dir) and len(os.listdir(chapters_dir)) > 0:
            process_chapters_directory(chapters_dir, chunks_dir, chunk_size, overlap)
        else:
            process_single_text_file(raw_text_file, chunks_dir, chunk_size, overlap)
    except Exception as e:
        print(f"预处理文本时出错: {e}")
        sys.exit(1)
    
    # 第三步：生成问答对
    print("\n===== 步骤3：生成问答对 (使用 OpenRouter) =====")
    
    # 检查 OpenRouter API 密钥是否存在
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("错误：OPENROUTER_API_KEY 环境变量未设置。")
        print("请在 .env 文件中定义 OPENROUTER_API_KEY='your_key' 或导出该环境变量。")
        sys.exit(1)
    
    qa_dir_param_for_generate_qa = dirs['qa_dataset']

    cmd = [
        f"python generate_qa.py {shlex.quote(chunks_dir)}",
        f"--output {shlex.quote(qa_dir_param_for_generate_qa)}",
        f"--model {model}",
        f"--questions {questions_per_chunk}",
        f"--sample {sample_ratio}",
    ]
    
    try:
        return_code = run_command(" ".join(cmd))
        if return_code != 0:
            print(f"生成问答对时出错，返回码: {return_code}")
            sys.exit(return_code)
    except Exception as e:
        print(f"执行生成问答对命令时出错: {e}")
        sys.exit(1)
    
    final_qa_dir = qa_dir_param_for_generate_qa
    
    summary = {
        "epub_file": os.path.basename(epub_path),
        "output_directory": output_dir,
        "raw_text_file": raw_text_file,
        "chapters_directory": chapters_dir,
        "chunks_directory": chunks_dir,
        "qa_dataset_directory": final_qa_dir,
        "processing_time": f"{(time.time() - start_time) / 60:.2f} 分钟",
        "parameters": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model": model,
            "questions_per_chunk": questions_per_chunk,
            "sample_ratio": sample_ratio,
            "api_provider": "OpenRouter"
        }
    }
    
    summary_file = os.path.join(output_dir, "processing_summary_openrouter.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理摘要已保存到: {summary_file}")
    print(f"\n整个流程已完成！总用时: {(time.time() - start_time) / 60:.2f} 分钟")
    print(f"所有输出文件都保存在: {output_dir}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行完整的经济学教材问答数据集构建流水线 (使用OpenRouter)")
    parser.add_argument("epub_path", help="EPUB文件的路径")
    parser.add_argument("--output", "-o", help="主输出目录", default=None)
    parser.add_argument("--no-chapters", action="store_false", dest="save_chapters", help="不保存单独的章节文件")
    parser.add_argument("--chunk-size", type=int, help="文本块的大小（单词数）", default=500)
    parser.add_argument("--overlap", type=int, help="文本块之间的重叠（单词数）", default=50)
    parser.add_argument("--model", "-m", help="OpenRouter上使用的模型名称", default="google/gemini-2.0-flash-001")
    parser.add_argument("--questions", "-q", type=int, help="每个块生成的问题数量", default=7)
    parser.add_argument("--sample", "-s", type=float, help="处理的文本块比例 (0-1)", default=1.0)
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            args.epub_path,
            args.output,
            args.save_chapters,
            args.chunk_size,
            args.overlap,
            args.model,
            args.questions,
            args.sample
        )
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        sys.exit(130)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1) 