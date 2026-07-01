from pathlib import Path
from typing import Any
import os

import gradio as gr

from src.pipeline import DEFAULT_DATA_ROOT, Pipeline, configs


DEFAULT_QUESTION = "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？"
EXAMPLE_QUESTIONS = [
    "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？",
    "中芯国际 2025 年一季度产能利用率和收入表现如何？",
    "请总结中芯国际未来两到三年的产能扩张计划。",
]
RETRIEVAL_MODE_LABELS = {
    "基础向量检索": "base",
    "整页上下文检索": "pdr",
    "智能重排检索": "max",
}
RETRIEVAL_MODE_DESCRIPTIONS = {
    "基础向量检索": "速度较快，适合简单事实查询。",
    "整页上下文检索": "返回完整页上下文，适合需要表格或段落上下文的问题。",
    "智能重排检索": "检索后再用模型重排，答案质量更稳，但耗时更长。",
}


def _normalize_config_name(config_name: str) -> str:
    """将前端中文检索模式转换为内部配置名称。"""
    return RETRIEVAL_MODE_LABELS.get(config_name, config_name)


def _get_config_label(config_name: str) -> str:
    """将内部配置名称转换为前端中文检索模式。"""
    for label, value in RETRIEVAL_MODE_LABELS.items():
        if value == config_name:
            return label
    return config_name


def _build_pipeline(config_name: str) -> Pipeline:
    """根据用户选择的配置创建问答流水线。"""
    normalized_config_name = _normalize_config_name(config_name)
    return Pipeline(DEFAULT_DATA_ROOT, run_config=configs[normalized_config_name])


def _friendly_error_message(exc: Exception) -> str:
    """将底层异常转换为前端用户可理解的中文提示。"""
    message = str(exc)
    if "DASHSCOPE_API_KEY" in message or "api_key" in message.lower():
        return "未检测到有效的 DashScope API Key，请先在 .env 中配置 DASHSCOPE_API_KEY。"
    if "No relevant context found" in message:
        return "没有检索到可用于回答的研报内容，请换一种问法或先确认向量库是否已生成。"
    if "No company name found" in message:
        return "没有识别到公司名称。当前数据集只有中芯国际，建议问题中明确写出“中芯国际”。"
    if "No matching vector DB" in message or "vector" in message.lower() and "found" in message.lower():
        return "未找到匹配的向量库，请先运行 process-reports 生成 FAISS 索引。"
    if "Connection" in message or "timeout" in message.lower():
        return "调用模型服务超时或网络连接失败，请检查网络和 DashScope 服务状态。"
    return f"生成答案时出错：{message}"


def _format_text(value: Any, empty_text: str = "暂无内容") -> str:
    """将模型返回内容转换为页面可展示文本。"""
    if value is None:
        return empty_text
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) or empty_text
    text = str(value).strip()
    return text if text else empty_text


def _format_markdown(value: Any, empty_text: str = "等待生成结果。") -> str:
    """将模型返回内容转换为 Markdown 文本。"""
    return _format_text(value, empty_text=empty_text)


def _format_status(text: str) -> str:
    """生成状态条 HTML。"""
    return f'<div class="status-strip">{text}</div>'


def _format_source_pages(answer: dict) -> list[list[Any]]:
    """整理原始文件名称和对应页码，便于 Gradio 表格展示。"""
    rows = []
    for ref in answer.get("references", []):
        rows.append([
            ref.get("file_name", ""),
            ref.get("page_index", ""),
            ref.get("pdf_sha1", ""),
        ])
    return rows


def generate_answer(question: str, config_name: str):
    """执行单问题检索问答，并返回页面展示内容。"""
    question = question.strip()
    if not question:
        raise gr.Error("请输入问题后再生成答案。")

    mode_label = _get_config_label(config_name)
    yield (
        _format_status(f"正在检索研报并生成答案，检索模式：{mode_label}"),
        "正在生成答案，请稍候。",
        [],
        "正在整理推理摘要。",
        "正在整理分步推理。",
    )

    try:
        pipeline = _build_pipeline(config_name)
        answer = pipeline.answer_single_question(question)
    except Exception as exc:
        raise gr.Error(_friendly_error_message(exc)) from exc

    status_text = _format_status(f"已完成检索与生成，检索模式：{mode_label}")
    final_answer = _format_markdown(answer.get("final_answer"))
    reasoning_summary = _format_markdown(answer.get("reasoning_summary"))
    step_by_step = _format_markdown(answer.get("step_by_step_analysis"))
    source_pages = _format_source_pages(answer)
    yield status_text, final_answer, source_pages, reasoning_summary, step_by_step


def clear_outputs():
    """清空结果区域。"""
    return _format_status("等待输入问题。"), "等待生成结果。", [], "等待生成结果。", "等待生成结果。"


CSS = """
.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    background: #f6f8fb;
    color: #182230;
}

.main-wrap {
    gap: 0 !important;
    min-height: 100vh;
}

.query-panel {
    min-height: 100vh;
    background: #edf2f7;
    padding: 28px 22px;
    border-right: 1px solid #d9e2ec;
}

.result-panel {
    padding: 28px 40px 36px 40px;
    max-width: 1280px;
}

.brand-block {
    margin-bottom: 22px;
}

.brand-block h1 {
    margin: 0;
    font-size: 24px;
    line-height: 1.25;
    color: #182230;
}

.brand-block p {
    margin: 8px 0 0 0;
    color: #667085;
    font-size: 14px;
}

.section-title {
    margin: 0 0 14px 0;
    font-size: 18px;
    font-weight: 700;
    color: #182230;
}

.field-label {
    margin: 14px 0 8px 0;
    color: #344054;
    font-size: 14px;
    font-weight: 700;
}

.sidebar-card {
    background: #ffffff;
    border: 1px solid #dfe7ef;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}

.sidebar-card .block,
.sidebar-card .form,
.sidebar-card .wrap,
.sidebar-card .html-container {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.sidebar-card textarea,
.sidebar-card input {
    background: #ffffff !important;
}

.topbar h1 {
    margin: 0;
    font-size: 30px;
    line-height: 1.2;
    color: #182230;
}

.status-strip {
    margin: 10px 0 22px 0;
    padding: 10px 12px;
    border: 1px solid #d6e4f5;
    border-radius: 8px;
    background: #eef6ff;
    color: #175cd3;
    font-size: 14px;
}

.result-label {
    margin: 0 0 10px 0;
    font-size: 16px;
    font-weight: 700;
    color: #182230;
}

.answer-card,
.summary-card,
.reasoning-card,
.source-card {
    background: #ffffff;
    border: 1px solid #dfe7ef;
    border-radius: 8px !important;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    margin-bottom: 16px;
}

.answer-card {
    border-left: 4px solid #2563eb;
}

.answer-card p,
.summary-card p,
.reasoning-card p,
.answer-card li,
.summary-card li,
.reasoning-card li {
    font-size: 16px !important;
    line-height: 1.7 !important;
    color: #344054;
}

.answer-card p {
    font-size: 18px !important;
    color: #182230;
}

.source-card .wrap {
    border: 0 !important;
}

.source-card table {
    font-size: 14px !important;
}

.primary-button {
    margin-top: 14px;
}

.primary-button button {
    height: 44px;
    border-radius: 8px !important;
    font-weight: 700 !important;
    background: #2563eb !important;
    border-color: #2563eb !important;
}

.clear-button button {
    height: 40px;
    border-radius: 8px !important;
}

.input-block textarea {
    min-height: 142px !important;
    line-height: 1.6 !important;
}

.compact-select {
    margin-bottom: 4px;
}

.mode-help {
    margin: 8px 0 12px 0;
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}

@media (max-width: 900px) {
    .query-panel {
        min-height: auto;
        border-right: 0;
        border-bottom: 1px solid #d9e2ec;
    }

    .result-panel {
        padding: 24px 18px 32px 18px;
    }
}
"""


with gr.Blocks(
    title="中芯国际研报 RAG 问答",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
        text_size="md",
    ),
    css=CSS,
) as demo:
    with gr.Row(elem_classes="main-wrap"):
        with gr.Column(scale=3, min_width=320, elem_classes="query-panel"):
            gr.HTML(
                """
                <div class="brand-block">
                <h1>研报问答</h1>
                <p>基于中芯国际研报的本地 RAG 检索与生成</p>
                </div>
                """,
            )
            with gr.Group(elem_classes="sidebar-card"):
                gr.HTML('<div class="section-title">查询设置</div>')
                gr.HTML('<div class="field-label">输入问题</div>')
                question_input = gr.Textbox(
                    label="",
                    value=DEFAULT_QUESTION,
                    lines=6,
                    max_lines=9,
                    placeholder="请输入关于中芯国际研报的问题",
                    show_copy_button=False,
                    show_label=False,
                    elem_classes="input-block",
                )
                gr.HTML('<div class="field-label">检索模式</div>')
                config_input = gr.Dropdown(
                    label="",
                    choices=list(RETRIEVAL_MODE_LABELS.keys()),
                    value="智能重排检索",
                    interactive=True,
                    show_label=False,
                    elem_classes="compact-select",
                )
                gr.HTML(
                    "<div class=\"mode-help\">"
                    "基础向量检索：速度较快。<br>"
                    "整页上下文检索：适合表格和长段落。<br>"
                    "智能重排检索：效果更稳，耗时更长。"
                    "</div>"
                )
                submit_button = gr.Button("生成答案", variant="primary", elem_classes="primary-button")
                clear_button = gr.Button("清空结果", variant="secondary", elem_classes="clear-button")
            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_input,
                label="示例问题",
            )

        with gr.Column(scale=11, min_width=640, elem_classes="result-panel"):
            gr.HTML('<div class="topbar"><h1>检索结果</h1></div>')
            status_output = gr.HTML(_format_status("等待输入问题。"))

            gr.Markdown("最终答案", elem_classes="result-label")
            final_output = gr.Markdown("等待生成结果。", elem_classes="answer-card")

            gr.Markdown("原始文件与页码", elem_classes="result-label")
            pages_output = gr.Dataframe(
                headers=["原始文件名称", "页码", "pdf_sha1"],
                datatype=["str", "number", "str"],
                row_count=(0, "dynamic"),
                col_count=(3, "fixed"),
                label="",
                interactive=False,
                wrap=True,
                elem_classes="source-card",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("推理摘要", elem_classes="result-label")
                    summary_output = gr.Markdown("等待生成结果。", elem_classes="summary-card")
                with gr.Column(scale=1):
                    gr.Markdown("分步推理", elem_classes="result-label")
                    step_output = gr.Markdown("等待生成结果。", elem_classes="reasoning-card")

    submit_button.click(
        fn=generate_answer,
        inputs=[question_input, config_input],
        outputs=[status_output, final_output, pages_output, summary_output, step_output],
    )
    question_input.submit(
        fn=generate_answer,
        inputs=[question_input, config_input],
        outputs=[status_output, final_output, pages_output, summary_output, step_output],
    )
    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[status_output, final_output, pages_output, summary_output, step_output],
    )


if __name__ == "__main__":
    server_port = os.getenv("GRADIO_SERVER_PORT")
    launch_kwargs = {"server_name": "127.0.0.1"}
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.queue(default_concurrency_limit=2).launch(**launch_kwargs)
