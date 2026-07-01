import json
from pathlib import Path
from typing import Optional
import pandas as pd
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文本分块工具类，只处理 MinerU 输出的 markdown 文件。
class TextSplitter():
    def _build_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """创建 LangChain 递归文本分割器，优先按 markdown 段落和换行切分。"""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True,
        )

    def _format_content_item(self, item: dict) -> str:
        """将 MinerU content_list 条目转换为可检索文本。"""
        item_type = item.get("type", "")
        if item_type in {"header", "footer"}:
            return ""
        if item.get("text"):
            return str(item["text"]).strip()
        if item.get("table_body"):
            return str(item["table_body"]).strip()
        if item.get("img_caption"):
            caption = item["img_caption"]
            if isinstance(caption, list):
                return "\n".join(str(text).strip() for text in caption if str(text).strip())
            return str(caption).strip()
        return ""

    def _estimate_line_range(self, source_text: str, chunk_text: str, search_start: int) -> tuple[list[int], int]:
        """根据分块文本在原文中的位置估算行号范围。"""
        start_char = source_text.find(chunk_text, search_start)
        if start_char < 0:
            start_char = search_start
        end_char = start_char + len(chunk_text)
        start_line = source_text.count("\n", 0, start_char) + 1
        end_line = source_text.count("\n", 0, end_char) + 1
        next_search_start = max(start_char + 1, end_char - 1)
        return [start_line, end_line], next_search_start

    def _split_page_text(self, page: int, page_text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
        """在单页内部使用 LangChain 分块，并保留页码和估算行号。"""
        splitter = self._build_splitter(chunk_size, chunk_overlap)
        documents = splitter.create_documents([page_text], metadatas=[{"page": page}])
        chunks = []
        search_start = 0
        for document in documents:
            chunk_text = document.page_content.strip()
            if not chunk_text:
                continue
            line_range, search_start = self._estimate_line_range(page_text, chunk_text, search_start)
            chunks.append({
                'page': document.metadata["page"],
                'lines': line_range,
                'text': chunk_text
            })
        return chunks

    def split_content_list_file(self, content_list_path: Path, chunk_size: int = 1200, chunk_overlap: int = 200):
        """
        按 MinerU content_list 的 page_idx 聚合页面，再用 LangChain 在页内分块。
        chunk_size 和 chunk_overlap 的单位为字符数。
        """
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)

        page_texts = {}
        for item in content_list:
            page_idx = item.get("page_idx")
            if page_idx is None:
                continue
            text = self._format_content_item(item)
            if not text:
                continue
            page = int(page_idx) + 1
            page_texts.setdefault(page, []).append(text)

        chunks = []
        pages = []
        for page in sorted(page_texts):
            page_text = "\n\n".join(page_texts[page])
            pages.append({"page": page, "text": page_text})
            chunks.extend(self._split_page_text(page, page_text, chunk_size, chunk_overlap))

        return chunks, pages

    def split_markdown_file(self, md_path: Path, chunk_size: int = 1200, chunk_overlap: int = 200):
        """
        使用 LangChain 分割 markdown 文件，每个分块记录估算起止行号和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大字符数
        :param chunk_overlap: 分块重叠字符数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        return self._split_page_text(1, markdown_text, chunk_size, chunk_overlap)

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 1200, chunk_overlap: int = 200, subset_csv: Optional[Path] = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大字符数
        :param chunk_overlap: 分块重叠字符数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for md_path in all_md_paths:
            content_list_path = md_path.with_name(f"{md_path.stem}_content_list.json")
            pages = []
            if content_list_path.exists():
                chunks, pages = self.split_content_list_file(content_list_path, chunk_size, chunk_overlap)
            else:
                chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks, "pages": pages}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name}")
        print(f"共分割 {len(all_md_paths)} 个 markdown 文件")
