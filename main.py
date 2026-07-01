import json

import click
from src.pipeline import Pipeline, configs, preprocess_configs, DEFAULT_DATA_ROOT

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def prepare_mineru_resources():
    """MinerU 使用本地模型，首次运行时由 MinerU 自行准备资源。"""
    Pipeline.prepare_mineru_resources()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel workers')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with MinerU."""
    pipeline = Pipeline(DEFAULT_DATA_ROOT)
    
    click.echo("Parsing PDFs with MinerU...")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(list(preprocess_configs.keys())), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    run_config = preprocess_configs[config]
    pipeline = Pipeline(DEFAULT_DATA_ROOT, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(list(configs.keys())), default='max', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    run_config = configs[config]
    pipeline = Pipeline(DEFAULT_DATA_ROOT, run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

@cli.command()
@click.argument("question", nargs=-1)
@click.option('--config', type=click.Choice(list(configs.keys())), default='max', help='Configuration preset to use')
@click.option('--json-output/--pretty-output', default=False, help='Print raw JSON result or readable text')
def ask(question, config, json_output):
    """Ask one question from command line."""
    question_text = " ".join(question).strip()
    if not question_text:
        question_text = click.prompt("请输入问题", type=str).strip()
    if not question_text:
        raise click.ClickException("问题不能为空。")

    run_config = configs[config]
    pipeline = Pipeline(DEFAULT_DATA_ROOT, run_config=run_config)
    answer = pipeline.answer_single_question(question_text)

    if json_output:
        click.echo(json.dumps(answer, ensure_ascii=False, indent=2))
        return

    click.echo("\n最终答案：")
    click.echo(answer.get("final_answer", ""))
    if answer.get("reasoning_summary"):
        click.echo("\n推理摘要：")
        click.echo(answer["reasoning_summary"])
    if answer.get("relevant_pages"):
        click.echo("\n相关页码：")
        click.echo(", ".join(str(page) for page in answer["relevant_pages"]))
    if answer.get("references"):
        click.echo("\n引用：")
        for ref in answer["references"]:
            click.echo(
                f"- file_name={ref.get('file_name', '')}, "
                f"page_index={ref.get('page_index', '')}, "
                f"pdf_sha1={ref.get('pdf_sha1', '')}"
            )

if __name__ == '__main__':
    cli()
