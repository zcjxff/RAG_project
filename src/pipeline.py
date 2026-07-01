from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

from src import pdf_mineru

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
DEFAULT_SOURCE_PDF_REPORTS_DIR = DEFAULT_DATA_ROOT / "pdf_reports"

class PipelineConfig:
    def __init__(
        self,
        root_path: Path,
        subset_name: str = "subset.csv",
        questions_file_name: str = "questions.json",
        pdf_reports_dir_name: Optional[Union[str, Path]] = None,
        serialized: bool = False,
        config_suffix: str = ""
    ):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = Path(pdf_reports_dir_name) if pdf_reports_dir_name else DEFAULT_SOURCE_PDF_REPORTS_DIR
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行数量，需要根据 DashScope 限流控制
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope"
    answering_model: str = "qwen-flash"
    config_suffix: str = ""

class Pipeline:
    def __init__(
        self,
        root_path: Optional[Path] = None,
        subset_name: str = "subset.csv",
        questions_file_name: str = "questions.json",
        pdf_reports_dir_name: Optional[Union[str, Path]] = None,
        run_config: RunConfig = RunConfig()
    ):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        root_path = Path(root_path) if root_path else DEFAULT_DATA_ROOT
        root_path.mkdir(parents=True, exist_ok=True)
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._ensure_subset_from_pdf_reports()
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: Optional[Union[str, Path]]) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _ensure_subset_from_pdf_reports(self):
        """
        根据 PDF 文件生成基础 subset.csv，提供 file_name、company_name、sha1 字段。
        """
        if self.paths.subset_path.exists():
            return

        pdf_paths = sorted(self.paths.pdf_reports_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"未找到 PDF 文件目录或目录为空: {self.paths.pdf_reports_dir}")

        rows = []
        for pdf_path in pdf_paths:
            with open(pdf_path, "rb") as file:
                sha1 = hashlib.sha1(file.read()).hexdigest()
            # 当前 stock_data 数据集均为中芯国际相关报告，统一公司名便于跨报告检索。
            company_name = "中芯国际"
            rows.append({
                "file_name": pdf_path.name,
                "company_name": company_name,
                "sha1": sha1
            })

        self.paths.subset_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.paths.subset_path, index=False, encoding="utf-8")

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def prepare_mineru_resources():
        # MinerU 首次运行时会自动准备本地资源
        print("MinerU will prepare local resources on first use.")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """兼容旧入口，使用 MinerU 本地并行解析全部 PDF。"""
        self.export_all_reports_to_markdown_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        print(f"开始处理: {file_name}")
        pdf_path = self.paths.pdf_reports_dir / file_name
        target_path = pdf_mineru.parse_pdf_to_markdown(pdf_path, self.paths.reports_markdown_path)
        print(f"已生成 markdown: {target_path}")

    def export_all_reports_to_markdown(self):
        """使用 MinerU 将全部 PDF 转换为 markdown。"""
        pdf_paths = sorted(self.paths.pdf_reports_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"未找到 PDF 文件: {self.paths.pdf_reports_dir}")

        for pdf_path in pdf_paths:
            self.export_reports_to_markdown(pdf_path.name)

    @staticmethod
    def _chunk_items(items: list, chunk_size: int) -> list:
        """按指定大小切分列表，便于分批并行处理。"""
        safe_chunk_size = max(1, chunk_size)
        return [items[index:index + safe_chunk_size] for index in range(0, len(items), safe_chunk_size)]

    def export_all_reports_to_markdown_parallel(self, chunk_size: int = 2, max_workers: int = 4):
        """
        使用 MinerU 并行解析全部 PDF。
        chunk_size 控制每个工作线程连续处理的 PDF 数量，max_workers 控制并发工作线程数。
        """
        pdf_paths = sorted(self.paths.pdf_reports_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"未找到 PDF 文件: {self.paths.pdf_reports_dir}")

        def process_batch(batch_paths: list[Path]):
            for pdf_path in batch_paths:
                self.export_reports_to_markdown(pdf_path.name)

        batches = self._chunk_items(pdf_paths, chunk_size)
        safe_max_workers = max(1, max_workers)
        with ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
            list(executor.map(process_batch, batches))

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        from src.text_splitter import TextSplitter

        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        from src.ingestion import VectorDBIngestor

        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 使用 MinerU 解析 PDF 报告
        if parallel:
            self.export_all_reports_to_markdown_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.export_all_reports_to_markdown()

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        from src.questions_processing import QuestionsProcessor

        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        from src.questions_processing import QuestionsProcessor

        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer

preprocess_configs = {"no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=4,
    submission_file=True,
    pipeline_details="MinerU PDF parsing + vector DB; llm = qwen-flash",
    answering_model="qwen-flash",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="MinerU PDF parsing + vector DB + parent document retrieval; llm = qwen-flash",
    answering_model="qwen-flash",
    config_suffix="_pdr"
)

max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="MinerU PDF parsing + vector DB + parent document retrieval + reranking; llm = qwen-flash",
    answering_model="qwen-flash",
    config_suffix="_qwen_flash"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置新项目的数据产物目录
    root_path = DEFAULT_DATA_ROOT
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)
    
    print('4. 将pdf转化为纯markdown文本')
    #pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf') 

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()     
    
    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('7. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions() 
    
    print('完成')
