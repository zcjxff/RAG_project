import json
import os
from pathlib import Path
from typing import List, Union

import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm


class DashScopeEmbeddings(Embeddings):
    def __init__(self, embedding_model: str = "text-embedding-v4"):
        load_dotenv()
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.embedding_model = embedding_model

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(4))
    def _get_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]:
        # 使用 DashScope text-embedding-v4 批量生成向量。
        text_chunks = text if isinstance(text, list) else [text]
        text_chunks = [chunk for chunk in text_chunks if isinstance(chunk, str) and chunk.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串。")

        embeddings = []
        max_batch_size = 10
        for index in range(0, len(text_chunks), max_batch_size):
            batch = text_chunks[index:index + max_batch_size]
            response = TextEmbedding.call(model=self.embedding_model, input=batch)
            output = getattr(response, "output", None)
            if output is None:
                raise RuntimeError(
                    "DashScope embedding API 未返回 output，"
                    f"code={getattr(response, 'code', None)}, "
                    f"message={getattr(response, 'message', None)}, "
                    f"request_id={getattr(response, 'request_id', None)}"
                )
            if "embeddings" not in output:
                raise RuntimeError(f"DashScope embedding API 返回格式异常: {response}")
            for item in output["embeddings"]:
                embedding = item.get("embedding")
                if not embedding:
                    raise RuntimeError("DashScope 返回的 embedding 为空。")
                embeddings.append(embedding)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings(text)[0]


class VectorDBIngestor:
    def __init__(self, embedding_model: str = "text-embedding-v4"):
        self.embeddings = DashScopeEmbeddings(embedding_model=embedding_model)

    def _existing_index_matches(self, output_path: Path, chunk_count: int) -> bool:
        if not output_path.exists():
            return False
        vector_store = FAISS.load_local(
            str(output_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store.index.ntotal == chunk_count

    def _process_report(self, report: dict):
        text_chunks = [chunk["text"][:2048] for chunk in report["content"]["chunks"] if chunk.get("text")]
        metadatas = [
            {
                "page": chunk.get("page", 0),
                "lines": chunk.get("lines", []),
            }
            for chunk in report["content"]["chunks"]
            if chunk.get("text")
        ]
        return FAISS.from_texts(
            texts=text_chunks,
            embedding=self.embeddings,
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = sorted(Path(all_reports_dir).glob("*.json"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports for FAISS"):
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)
            sha1 = report_data["metainfo"].get("sha1", "")
            if not sha1:
                raise ValueError(f"分块报告 {report_path} 缺少 sha1 字段，无法保存 faiss 文件。")
            output_path = output_dir / sha1
            chunk_count = len([chunk for chunk in report_data["content"]["chunks"] if chunk.get("text")])
            if self._existing_index_matches(output_path, chunk_count):
                continue
            vector_store = self._process_report(report_data)
            vector_store.save_local(str(output_path))

        print(f"Processed {len(all_report_paths)} reports")
