import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from src.ingestion import DashScopeEmbeddings
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)


def _tokenize_for_bm25(text: str) -> List[str]:
    """面向中文研报的 BM25 分词：中文按字切分，英文数字按连续片段保留。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.%+-]+", text.lower())


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_model: str = "text-embedding-v4"):
        self.vector_db_dir = Path(vector_db_dir)
        self.documents_dir = Path(documents_dir)
        self.embedding_model = embedding_model
        self.embeddings = DashScopeEmbeddings(embedding_model=embedding_model)
        self.all_dbs = self._load_dbs()

    def _get_embedding(self, text: str):
        return self.embeddings.embed_query(text)

    def _load_dbs(self):
        all_dbs = []
        for document_path in self.documents_dir.glob("*.json"):
            with open(document_path, "r", encoding="utf-8") as file:
                document = json.load(file)
            sha1 = document.get("metainfo", {}).get("sha1")
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / sha1
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
            all_dbs.append({
                "name": sha1,
                "vector_db": FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                ),
                "document": document,
            })
        return all_dbs

    def _find_reports_by_company_name(self, company_name: str):
        reports = []
        for report in self.all_dbs:
            metainfo = report.get("document", {}).get("metainfo", {})
            if metainfo.get("company_name") == company_name or company_name in metainfo.get("file_name", ""):
                reports.append(report)
        if not reports:
            raise ValueError(f"未找到公司相关报告: {company_name}")
        return reports

    @staticmethod
    def _get_source_file_name(document: dict) -> str:
        """将分块文件中的 markdown 文件名还原为原始 PDF 文件名。"""
        file_name = document["metainfo"]["file_name"]
        if Path(file_name).suffix == ".md":
            return Path(file_name).with_suffix(".pdf").name
        return file_name

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,
        top_n: int = 3,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        reports = self._find_reports_by_company_name(company_name)
        query_embedding = self._get_embedding(query)
        retrieval_results = []

        for report in reports:
            document = report["document"]
            vector_store = report["vector_db"]
            chunks = document["content"]["chunks"]
            pages = document["content"].get("pages", [])
            actual_top_n = min(top_n, len(chunks))
            scored_documents = vector_store.similarity_search_with_score_by_vector(
                query_embedding,
                k=actual_top_n,
            )
            seen_pages = set()

            for retrieved_document, distance in scored_documents:
                page_number = retrieved_document.metadata.get("page", 0)
                parent_page = next((page for page in pages if page.get("page") == page_number), None)
                if return_parent_pages and parent_page:
                    if page_number in seen_pages:
                        continue
                    seen_pages.add(page_number)
                    text = parent_page["text"]
                else:
                    text = retrieved_document.page_content
                retrieval_results.append({
                    "distance": round(float(distance), 4),
                    "page": page_number,
                    "text": text,
                    "pdf_sha1": document["metainfo"].get("sha1", ""),
                    "file_name": self._get_source_file_name(document),
                })

        retrieval_results.sort(key=lambda item: item["distance"], reverse=True)
        return retrieval_results[:top_n]

    def retrieve_all(self, company_name: str) -> List[Dict]:
        reports = self._find_reports_by_company_name(company_name)
        all_chunks = []
        for report in reports:
            document = report["document"]
            for chunk in document["content"]["chunks"]:
                all_chunks.append({
                    "distance": 0.0,
                    "page": chunk.get("page", 0),
                    "text": chunk["text"],
                    "pdf_sha1": document["metainfo"].get("sha1", ""),
                    "file_name": self._get_source_file_name(document),
                })
        return all_chunks


class BM25ReportRetriever:
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self.all_dbs = self._load_dbs()

    def _load_dbs(self):
        all_dbs = []
        for document_path in self.documents_dir.glob("*.json"):
            with open(document_path, "r", encoding="utf-8") as file:
                document = json.load(file)
            chunks = document.get("content", {}).get("chunks", [])
            pages = document.get("content", {}).get("pages", [])
            bm25_documents = []
            for chunk_index, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                if not text:
                    continue
                page_number = chunk.get("page", 0)
                parent_page = next((page for page in pages if page.get("page") == page_number), None)
                bm25_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "chunk_index": chunk_index,
                            "page": page_number,
                            "lines": chunk.get("lines", []),
                            "parent_text": parent_page["text"] if parent_page else "",
                            "pdf_sha1": document["metainfo"].get("sha1", ""),
                            "file_name": VectorRetriever._get_source_file_name(document),
                        },
                    )
                )
            if not bm25_documents:
                continue
            all_dbs.append({
                "document": document,
                "documents": bm25_documents,
            })
        return all_dbs

    def _find_reports_by_company_name(self, company_name: str):
        reports = []
        for report in self.all_dbs:
            metainfo = report.get("document", {}).get("metainfo", {})
            if metainfo.get("company_name") == company_name or company_name in metainfo.get("file_name", ""):
                reports.append(report)
        if not reports:
            raise ValueError(f"未找到公司相关报告: {company_name}")
        return reports

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 10,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        reports = self._find_reports_by_company_name(company_name)
        candidate_documents = []
        for report in reports:
            candidate_documents.extend(report["documents"])
        retriever = BM25Retriever.from_documents(
            candidate_documents,
            preprocess_func=_tokenize_for_bm25,
        )
        bm25_documents = retriever.vectorizer.get_top_n(
            retriever.preprocess_func(query),
            retriever.docs,
            n=top_n,
        )
        retrieval_results = []
        seen_pages = set()

        for rank, bm25_document in enumerate(bm25_documents, start=1):
            page_number = bm25_document.metadata.get("page", 0)
            pdf_sha1 = bm25_document.metadata.get("pdf_sha1", "")
            if return_parent_pages and bm25_document.metadata.get("parent_text"):
                page_key = (pdf_sha1, page_number)
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                text = bm25_document.metadata["parent_text"]
            else:
                text = bm25_document.page_content
            retrieval_results.append({
                "distance": round(1 / rank, 4),
                "bm25_rank": rank,
                "page": page_number,
                "text": text,
                "pdf_sha1": pdf_sha1,
                "file_name": bm25_document.metadata.get("file_name", ""),
                "retrieval_source": "bm25",
            })

        retrieval_results.sort(key=lambda item: item["distance"], reverse=True)
        return retrieval_results[:top_n]


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.bm25_retriever = BM25ReportRetriever(documents_dir)
        self.reranker = LLMReranker()

    @staticmethod
    def _merge_retrieval_results(vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        merged_results = {}
        for result in vector_results:
            key = (result.get("pdf_sha1", ""), result.get("page", 0), result.get("text", ""))
            result_with_source = result.copy()
            result_with_source["retrieval_source"] = "vector"
            merged_results[key] = result_with_source

        for result in bm25_results:
            key = (result.get("pdf_sha1", ""), result.get("page", 0), result.get("text", ""))
            if key not in merged_results:
                merged_results[key] = result.copy()
                continue
            existing = merged_results[key]
            existing["retrieval_source"] = "vector,bm25"
            existing["bm25_rank"] = result.get("bm25_rank")
            existing["distance"] = max(existing.get("distance", 0.0), result.get("distance", 0.0))

        return sorted(merged_results.values(), key=lambda item: item["distance"], reverse=True)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
        )
        bm25_results = self.bm25_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
        )
        candidate_results = self._merge_retrieval_results(vector_results, bm25_results)
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=candidate_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )
        return reranked_results[:top_n]

    def retrieve_all(self, company_name: str) -> List[Dict]:
        return self.vector_retriever.retrieve_all(company_name)
