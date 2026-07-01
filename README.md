# rag_stock_qwen

这是一个面向中芯国际研报的本地 RAG 问答项目。项目使用 MinerU 本地解析 PDF，使用 DashScope `text-embedding-v4` 生成向量，使用 LangChain FAISS 和 BM25 做检索，并通过 Qwen 模型生成结构化答案。前端使用 Gradio 搭建，页面布局参考 `demo_cn/UI界面参考-完成.png`。

## 当前完成情况

- 已完成 PDF 到 Markdown 的解析入口。
- 已完成 Markdown 和 MinerU `content_list.json` 分块，分块逻辑使用 LangChain `RecursiveCharacterTextSplitter`。
- 已完成 LangChain FAISS 向量库构建。
- 已完成 BM25 + 向量检索的混合召回，并接入现有 LLM 重排流程。
- 已完成单问题 CLI 问答。
- 已完成 Gradio 本地前端页面。
- 已补充 `.env.example`、`.gitignore` 和批量问题样例 `data/stock_data/questions.json`。
- 当前项目数据目录已有 9 份原始 PDF、9 份 Markdown、9 份分块 JSON、9 个 LangChain FAISS 索引。

## 目录说明

```text
rag_stock_qwen/
├── app_gradio.py              # Gradio 前端入口
├── main.py                    # Click 命令行入口
├── requirements.txt           # Python 依赖
├── run_python.sh              # macOS 本机 Python 运行脚本
├── setup.py                   # 包配置
├── src/
│   ├── pipeline.py            # 主流程编排
│   ├── pdf_mineru.py          # MinerU PDF 解析
│   ├── text_splitter.py       # 文档分块
│   ├── ingestion.py           # embedding 与 FAISS 入库
│   ├── retrieval.py           # 向量检索和混合检索
│   ├── reranking.py           # LLM 重排
│   ├── api_requests.py        # DashScope API 调用
│   └── questions_processing.py # 问题处理和答案格式化
└── data/stock_data/
    ├── pdf_reports/          # 原始 PDF 文件
    ├── subset.csv
    ├── debug_data/03_reports_markdown/
    └── databases/
        ├── chunked_reports/  # 分块 JSON
        └── vector_dbs/       # LangChain FAISS 索引目录
```

PDF 原始文件默认读取项目内目录：

```text
data/stock_data/pdf_reports/
```

项目不再默认依赖工作区根目录的 `../data/stock_data/pdf_reports/`。如果重新解析 PDF，会直接从项目内 `data/stock_data/pdf_reports/` 获取原始文件。

## 环境准备

本机是 macOS，项目要求 Python `3.13.12`。可以使用项目自带脚本运行系统 Python：

```bash
cd /Users/cjz/Desktop/实战项目/RAG/rag_stock_qwen
./run_python.sh --version
```

安装依赖：

```bash
cd /Users/cjz/Desktop/实战项目/RAG/rag_stock_qwen
./run_python.sh -m pip install -r requirements.txt
```

配置 DashScope Key。推荐先复制示例文件：

```bash
cp .env.example .env
```

然后在项目根目录的 `.env` 中填写：

```bash
DASHSCOPE_API_KEY=你的DashScope API Key
```

也可以在当前 shell 中临时设置：

```bash
export DASHSCOPE_API_KEY=你的DashScope API Key
```

MinerU 当前使用本地解析模式，不需要配置 MinerU 云端 token。DashScope Key 用于 embedding、Qwen 生成和 LLM 重排。

## 启动 Gradio 前端

默认启动：

```bash
cd /Users/cjz/Desktop/实战项目/RAG/rag_stock_qwen
./run_python.sh app_gradio.py
```

如果默认端口被占用，可以指定端口：

```bash
GRADIO_SERVER_PORT=64885 ./run_python.sh app_gradio.py
```

启动后浏览器访问终端输出的本地地址，例如：

```text
http://127.0.0.1:64885
```

页面支持输入问题、选择检索模式，并展示：

- 分步推理
- 推理摘要
- 原始文件名称和对应页码
- 最终答案

## 命令行用法

查看命令：

```bash
./run_python.sh main.py --help
```

解析 PDF：

```bash
./run_python.sh main.py parse-pdfs
```

顺序解析 PDF：

```bash
./run_python.sh main.py parse-pdfs --sequential
```

并行解析 PDF：

```bash
./run_python.sh main.py parse-pdfs --parallel --chunk-size 2 --max-workers 4
```

处理已解析报告，生成分块 JSON 和 LangChain FAISS 索引：

```bash
./run_python.sh main.py process-reports
```

`process-reports` 会调用 DashScope embedding API，因此需要有效 `DASHSCOPE_API_KEY` 和网络连接。生成后的向量库是 LangChain FAISS 目录结构：

```text
data/stock_data/databases/vector_dbs/<pdf_sha1>/index.faiss
data/stock_data/databases/vector_dbs/<pdf_sha1>/index.pkl
```

BM25 不需要单独生成索引文件，会在检索时基于 `data/stock_data/databases/chunked_reports/` 中的分块 JSON 构建。

单问题问答：

```bash
./run_python.sh main.py ask "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？"
```

输出 JSON：

```bash
./run_python.sh main.py ask "中芯国际的产能利用率如何？" --json-output
```

批量问题处理会读取 `data/stock_data/questions.json`：

```bash
./run_python.sh main.py process-questions
```

## 检索模式

`main.py` 和 Gradio 页面共用 `src/pipeline.py` 中的配置。Gradio 页面显示中文名称，内部会自动映射到对应配置：

- `基础向量检索`：对应 `base`，只使用 LangChain FAISS 向量检索。
- `整页上下文检索`：对应 `pdr`，使用 LangChain FAISS 向量检索并返回整页上下文。
- `智能重排检索`：对应 `max`，使用 LangChain FAISS 向量检索 + LangChain BM25 检索做混合召回，再使用 LLM 重排。

默认使用 `智能重排检索`。

混合检索流程：

```text
问题
-> DashScope embedding + LangChain FAISS 向量召回
-> LangChain BM25 词面召回
-> 合并去重
-> LLMReranker 重排
-> 返回最终上下文
```

## 注意事项

- Gradio 页面只负责调用现有 RAG 流程，不会重新解析 PDF 或重新生成向量库。
- 生成答案时会调用 DashScope embedding、Qwen 模型和可选 LLM 重排，需要网络和有效 `DASHSCOPE_API_KEY`。
- 如果只想测试界面是否能打开，可以先启动 Gradio，不点击“生成答案”。
- 当前已提供 `data/stock_data/questions.json` 样例，可以直接替换或追加自己的问题。
- `parse-pdfs` 会调用本机 MinerU 解析 PDF，首次运行可能会准备本地模型资源。
- 如果新增或替换 PDF，建议按顺序执行 `parse-pdfs`、`process-reports`，再启动问答流程。
