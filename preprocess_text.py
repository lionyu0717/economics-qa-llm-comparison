#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对提取的文本进行预处理和分块
使用方法：python preprocess_text.py path/to/extracted_text.txt
"""

import sys
import os
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# 下载NLTK资源（如果没有）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载NLTK punkt tokenizer...")
    nltk.download('punkt')

def clean_text(text):
    """清理文本，移除不必要的内容"""
    # 移除页码（通常是页面底部或顶部的数字）
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)
    
    # 替换连续的换行符为单个换行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 替换连续的空格为单个空格
    text = re.sub(r' {2,}', ' ', text)
    
    # 移除奇怪的符号或控制字符
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    return text.strip()

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    将文本分割成固定大小的块，保留句子边界
    
    参数:
        text: 要分割的文本
        chunk_size: 每个块的目标大小（单词数）
        overlap: 相邻块之间的重叠单词数
    
    返回:
        包含分块文本的列表
    """
    # 分割成句子
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # 计算句子中的单词数
        sentence_words = len(sentence.split())
        
        # 如果当前块加上这个句子超过了目标大小，保存当前块并开始新块
        if current_size + sentence_words > chunk_size and current_chunk:
            # 将当前块合并成文本
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # 新块从前一个块的最后几个句子开始（实现重叠）
            overlap_words = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_words + s_words <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_words
        
        # 添加当前句子到块
        current_chunk.append(sentence)
        current_size += sentence_words
    
    # 添加最后的块
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def process_chapters_directory(chapters_dir, output_dir=None, chunk_size=500, overlap=50):
    """处理章节目录中的所有文本文件并分块"""
    if not os.path.isdir(chapters_dir):
        raise ValueError(f"指定的章节目录不存在: {chapters_dir}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = f"{chapters_dir}_chunked"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.txt文件（跳过index文件）
    txt_files = [f for f in os.listdir(chapters_dir) 
                if f.endswith('.txt') and f != 'chapters_index.txt']
    txt_files.sort()  # 确保按顺序处理
    
    all_chunks = []
    chunk_mapping = {}
    
    print(f"处理目录中的 {len(txt_files)} 个章节文件...")
    
    for file_name in tqdm(txt_files, desc="处理章节"):
        file_path = os.path.join(chapters_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read()
            
            # 从文件名中提取章节编号和标题
            match = re.match(r'(\d+)_(.+)\.txt', file_name)
            if match:
                chapter_num = match.group(1)
                chapter_title = match.group(2)
            else:
                chapter_num = txt_files.index(file_name) + 1
                chapter_title = os.path.splitext(file_name)[0]
            
            # 清理文本
            cleaned_text = clean_text(chapter_text)
            
            # 分块
            chapter_chunks = split_text_into_chunks(cleaned_text, chunk_size, overlap)
            
            # 为每个块添加元数据并保存
            for i, chunk in enumerate(chapter_chunks):
                chunk_id = f"{chapter_num}_{i+1:03d}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "chapter": chapter_title,
                    "chapter_num": chapter_num,
                    "chunk_num": i+1,
                    "text": chunk
                }
                
                all_chunks.append(chunk_data)
                chunk_mapping[chunk_id] = len(all_chunks) - 1
                
                # 保存单独的块文件
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_id}.json")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    # 保存所有块的索引
    index_file = os.path.join(output_dir, "chunks_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
            "chunk_mapping": chunk_mapping
        }, f, ensure_ascii=False, indent=2)
    
    print(f"总共处理了 {len(all_chunks)} 个文本块，保存到目录 {output_dir}")
    print(f"文本块索引已保存到 {index_file}")
    
    return all_chunks

def process_single_text_file(file_path, output_dir=None, chunk_size=500, overlap=50):
    """处理单个文本文件并分块"""
    if not os.path.isfile(file_path):
        raise ValueError(f"指定的文本文件不存在: {file_path}")
    
    # 设置输出目录
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = f"{base_name}_chunked"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用分隔符分割章节
        chapters = re.split(r'\n\n={50}\n(.+?)\n={50}\n\n', text)
        
        if len(chapters) <= 1:
            # 如果没有找到章节分隔符，将整个文本作为一个大章节
            print("警告：未检测到章节分隔符，将整个文本作为一个章节处理")
            chapters = ["整本书", text]
        
        all_chunks = []
        chunk_mapping = {}
        chapter_texts = []
        
        # 处理章节（注意，分割结果中，偶数索引是标题，奇数索引是内容）
        for i in range(0, len(chapters) - 1, 2):
            if i + 1 < len(chapters):
                chapter_title = chapters[i+1].strip()
                chapter_text = chapters[i+2].strip() if i+2 < len(chapters) else ""
                chapter_texts.append((chapter_title, chapter_text))
        
        # 如果上面的分割失败，尝试另一种方式
        if not chapter_texts:
            print("尝试使用另一种方式分割章节...")
            chapters = re.split(r'\n\n={50}\n(.+?)\n={50}\n\n', text)
            chapter_titles = [m.group(1) for m in re.finditer(r'\n\n={50}\n(.+?)\n={50}\n\n', text)]
            chapter_contents = chapters
            
            if len(chapter_titles) == len(chapter_contents):
                for i in range(len(chapter_titles)):
                    chapter_texts.append((chapter_titles[i], chapter_contents[i]))
            else:
                # 如果还是失败，将整个文本作为一个章节
                chapter_texts = [("整本书", text)]
        
        print(f"从文本中提取了 {len(chapter_texts)} 个章节")
        
        # 处理每个章节
        for chapter_num, (chapter_title, chapter_text) in enumerate(tqdm(chapter_texts, desc="处理章节")):
            # 清理文本
            cleaned_text = clean_text(chapter_text)
            
            # 分块
            chapter_chunks = split_text_into_chunks(cleaned_text, chunk_size, overlap)
            
            # 为每个块添加元数据并保存
            for i, chunk in enumerate(chapter_chunks):
                chunk_id = f"{chapter_num+1:02d}_{i+1:03d}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "chapter": chapter_title,
                    "chapter_num": chapter_num + 1,
                    "chunk_num": i + 1,
                    "text": chunk
                }
                
                all_chunks.append(chunk_data)
                chunk_mapping[chunk_id] = len(all_chunks) - 1
                
                # 保存单独的块文件
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_id}.json")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        # 保存所有块的索引
        index_file = os.path.join(output_dir, "chunks_index.json")
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
                "chunk_mapping": chunk_mapping
            }, f, ensure_ascii=False, indent=2)
        
        print(f"总共处理了 {len(all_chunks)} 个文本块，保存到目录 {output_dir}")
        print(f"文本块索引已保存到 {index_file}")
        
        return all_chunks
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        raise

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("用法: python preprocess_text.py path/to/text_file_or_chapters_dir [output_dir] [--chunk-size=500] [--overlap=50]")
            sys.exit(1)
        
        source_path = sys.argv[1]
        
        # 解析参数
        output_dir = None
        chunk_size = 500
        overlap = 50
        
        for arg in sys.argv[2:]:
            if arg.startswith('--chunk-size='):
                chunk_size = int(arg.split('=')[1])
            elif arg.startswith('--overlap='):
                overlap = int(arg.split('=')[1])
            elif not arg.startswith('--'):
                output_dir = arg
        
        # 根据源路径是文件还是目录选择处理方法
        if os.path.isdir(source_path):
            process_chapters_directory(source_path, output_dir, chunk_size, overlap)
        else:
            process_single_text_file(source_path, output_dir, chunk_size, overlap)
        
        print("文本预处理和分块完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1) 