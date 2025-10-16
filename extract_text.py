#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从EPUB文件中提取文本
使用方法：python extract_text.py path/to/epub_file.epub
"""

import sys
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

def clean_text(text):
    """清理HTML提取的文本，移除多余空白符和特殊字符"""
    # 替换多个空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除各种控制字符
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text.strip()

def html_to_text(html_content):
    """将HTML内容转换为纯文本"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # 移除脚本和样式元素
        for script in soup(['script', 'style']):
            script.extract()
        # 获取文本
        text = soup.get_text()
        # 清理文本
        return clean_text(text)
    except Exception as e:
        print(f"HTML解析错误: {e}")
        return ""

def extract_chapter_info(item):
    """尝试从epub项目中提取章节信息"""
    try:
        html_content = item.get_content().decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html_content, 'html.parser')
        # 尝试找出章节标题，这里的选择器可能需要根据具体EPUB调整
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        if headers:
            return headers[0].get_text().strip()
        return item.get_name()  # 如果找不到标题，使用文件名
    except:
        return item.get_name()  # 出错时使用文件名

def epub_to_text(epub_path, output_txt=None, save_chapters=False):
    """从EPUB文件中提取文本
    
    参数:
        epub_path: EPUB文件路径
        output_txt: 输出文本文件路径，如果为None，将使用EPUB文件名作为基础
        save_chapters: 是否将每个章节保存为单独的文件
    
    返回:
        提取的文本内容
    """
    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"EPUB文件 '{epub_path}' 未找到")

    # 如果没有指定输出文件，根据EPUB文件名生成
    if output_txt is None:
        base_name = os.path.splitext(os.path.basename(epub_path))[0]
        output_txt = f"{base_name}_extracted.txt"
    
    # 创建用于保存章节的目录
    if save_chapters:
        chapters_dir = f"{os.path.splitext(os.path.basename(epub_path))[0]}_chapters"
        os.makedirs(chapters_dir, exist_ok=True)

    print(f"正在解析EPUB文件: {epub_path}")
    book = epub.read_epub(epub_path)
    print(f"文件已加载，开始提取内容...")

    # 按顺序处理所有文档项目
    all_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    full_text = []
    chapters = []

    for i, item in enumerate(tqdm(all_items, desc="提取章节")):
        try:
            content = item.get_content().decode('utf-8', errors='ignore')
            chapter_text = html_to_text(content)
            
            if not chapter_text.strip():  # 跳过空章节
                continue
                
            chapter_title = extract_chapter_info(item)
            chapter_info = f"\n\n{'='*50}\n{chapter_title}\n{'='*50}\n\n"
            
            # 将章节标题和内容添加到总文本
            full_text.append(chapter_info + chapter_text)
            
            # 保存单独的章节文件
            if save_chapters:
                # 清理章节标题以用于文件名，移除特殊字符
                safe_title_part = re.sub(r'[^\w\s-]', '', chapter_title)
                chapter_filename = f"{i+1:03d}_{safe_title_part[:50]}.txt"
                chapter_path = os.path.join(chapters_dir, chapter_filename)
                with open(chapter_path, 'w', encoding='utf-8') as f:
                    f.write(chapter_text)
                chapters.append((chapter_title, chapter_path))
                print(f"已保存章节: {chapter_filename}")
        except Exception as e:
            print(f"处理章节 {item.get_name()} 时出错: {e}")

    # 合并所有文本
    combined_text = "\n".join(full_text)
    
    # 保存完整文本
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print(f"所有文本已提取并保存到: {output_txt}")
    if save_chapters:
        print(f"单独的章节文件已保存到目录: {chapters_dir}")
        # 创建章节索引文件
        with open(os.path.join(chapters_dir, "chapters_index.txt"), 'w', encoding='utf-8') as f:
            for i, (title, path) in enumerate(chapters):
                f.write(f"{i+1:03d}. {title} -> {os.path.basename(path)}\n")
        print(f"章节索引已保存到: {os.path.join(chapters_dir, 'chapters_index.txt')}")
    
    return combined_text

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("用法: python extract_text.py path/to/epub_file.epub [output_file.txt] [--save-chapters]")
            sys.exit(1)
        
        epub_path = sys.argv[1]
        output_txt = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
        save_chapters = '--save-chapters' in sys.argv
        
        epub_to_text(epub_path, output_txt, save_chapters)
        print("文本提取完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1) 