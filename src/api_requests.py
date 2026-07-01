import json
import os
from typing import Optional, Type

import dashscope
from dotenv import load_dotenv
from pydantic import BaseModel

import src.prompts as prompts


class DashscopeProcessor:
    def __init__(self, default_model: str = "qwen-flash"):
        load_dotenv()
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = default_model
        self.response_data = {}

    def send_message(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        response_format: Optional[Type[BaseModel]] = None,
    ):
        # 调用 DashScope Qwen，并在需要时按 Pydantic 模型校验 JSON 输出。
        model = model or self.default_model
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ]
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format="message",
        )
        content = response.output.choices[0].message.content
        self.response_data = {
            "model": model,
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }

        if response_format is None:
            return content

        json_text = self._extract_json_text(content)
        data = json.loads(json_text)
        return response_format.model_validate(data).model_dump()

    @staticmethod
    def _extract_json_text(content: str) -> str:
        content = content.strip()
        if content.startswith("```") and "```" in content[3:]:
            first_backtick = content.find("```") + 3
            next_newline = content.find("\n", first_backtick)
            if next_newline > 0:
                first_backtick = next_newline + 1
            last_backtick = content.rfind("```")
            return content[first_backtick:last_backtick].strip()
        return content


class APIProcessor:
    def __init__(self, provider: str = "dashscope"):
        if provider.lower() != "dashscope":
            raise ValueError("当前项目仅支持 dashscope")
        self.processor = DashscopeProcessor()
        self.response_data = {}

    def get_answer_from_rag_context(self, question: str, rag_context: str, schema: str, model: str):
        system_prompt = prompts.AnswerWithRAGContextStringPrompt.system_prompt
        user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            response_format=prompts.AnswerWithRAGContextStringPrompt.AnswerSchema,
        )
        self.response_data = self.processor.response_data
        return answer_dict

    def get_reranking_scores(self, query: str, documents: list[str], model: str = "qwen-flash"):
        user_prompt = prompts.RerankingPrompt.user_prompt.format(
            query=query,
            documents="\n\n---\n\n".join(
                f'Block {index + 1}:\n"""\n{text}\n"""'
                for index, text in enumerate(documents)
            ),
            count=len(documents),
        )
        result = self.processor.send_message(
            model=model,
            temperature=0,
            system_content=prompts.RerankingPrompt.system_prompt,
            human_content=user_prompt,
            response_format=prompts.RetrievalRankingMultipleBlocks,
        )
        self.response_data = self.processor.response_data
        return result
