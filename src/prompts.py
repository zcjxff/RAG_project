from typing import List

from pydantic import BaseModel, Field


class AnswerWithRAGContextStringPrompt:
    system_prompt = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于检索到的中芯国际相关报告内容回答问题。
不要使用外部知识，不要编造信息。
如果上下文没有足够信息，请在 final_answer 中明确说明未找到相关依据。

你的回答必须是 JSON，字段固定为：
- step_by_step_analysis: 字符串，说明你如何基于上下文判断答案
- reasoning_summary: 字符串，简要总结依据
- relevant_pages: 数字列表，只包含直接支持答案的页码；如果上下文页码为0则使用0
- final_answer: 字符串，直接回答用户问题
"""

    user_prompt = """
以下是上下文:
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="基于上下文的推理过程。")
        reasoning_summary: str = Field(description="简要总结答案依据。")
        relevant_pages: List[int] = Field(description="直接支持答案的页码。")
        final_answer: str = Field(description="最终答案，必须是字符串。")


class RerankingPrompt:
    system_prompt = """
你是一个RAG检索重排专家。
你将收到一个查询和若干检索到的文本块，请分别对每个块与查询的相关性进行评分。

评分范围为0到1：
- 0 表示完全无关
- 1 表示完全相关

你的回答必须是 JSON，字段固定为：
{
  "block_rankings": [
    {"reasoning": "评分理由", "relevance_score": 0.8}
  ]
}
block_rankings 的数量必须与输入文本块数量一致，并保持原顺序。
"""

    user_prompt = """
查询：
"{query}"

文本块：
{documents}

请按原顺序输出恰好 {count} 个评分。
"""


class RetrievalRankingSingleBlock(BaseModel):
    reasoning: str = Field(description="相关性评分理由。")
    relevance_score: float = Field(description="相关性分数，范围0到1。")


class RetrievalRankingMultipleBlocks(BaseModel):
    block_rankings: List[RetrievalRankingSingleBlock] = Field(description="每个文本块的评分列表。")
