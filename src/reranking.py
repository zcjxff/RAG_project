from concurrent.futures import ThreadPoolExecutor

from src.api_requests import APIProcessor


class LLMReranker:
    def __init__(self):
        self.api_processor = APIProcessor(provider="dashscope")

    def get_rank_for_multiple_blocks(self, query: str, retrieved_documents: list[str]):
        return self.api_processor.get_reranking_scores(query=query, documents=retrieved_documents)

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        doc_batches = [documents[index:index + documents_batch_size] for index in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight

        def process_batch(batch):
            texts = [doc["text"] for doc in batch]
            rankings = self.get_rank_for_multiple_blocks(query, texts).get("block_rankings", [])
            if len(rankings) != len(batch):
                raise ValueError(f"重排结果数量不匹配，期望 {len(batch)}，实际 {len(rankings)}。")

            results = []
            for doc, rank in zip(batch, rankings):
                score = float(rank["relevance_score"])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = score
                doc_with_score["combined_score"] = round(
                    llm_weight * score + vector_weight * doc["distance"],
                    4,
                )
                results.append(doc_with_score)
            return results

        with ThreadPoolExecutor(max_workers=1) as executor:
            batch_results = list(executor.map(process_batch, doc_batches))

        all_results = []
        for batch in batch_results:
            all_results.extend(batch)
        all_results.sort(key=lambda item: item["combined_score"], reverse=True)
        return all_results
