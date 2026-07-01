import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """
    使用 MinerU 本地 CLI 将单个 PDF 解析为 markdown。
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{pdf_path.stem}.md"
    content_list_path = output_dir / f"{pdf_path.stem}_content_list.json"
    if target_path.exists() and content_list_path.exists():
        return target_path

    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        command = [
            "mineru",
            "-p",
            str(pdf_path),
            "-o",
            str(tmp_path),
            "-b",
            "pipeline",
        ]
        subprocess.run(command, check=True)

        markdown_paths = sorted(tmp_path.rglob("*.md"))
        if not markdown_paths:
            raise FileNotFoundError(f"MinerU 未生成 markdown 文件: {pdf_path}")

        shutil.copy2(markdown_paths[0], target_path)
        content_list_paths = sorted(tmp_path.rglob("*_content_list.json"))
        if not content_list_paths:
            raise FileNotFoundError(f"MinerU 未生成 content_list 文件: {pdf_path}")
        shutil.copy2(content_list_paths[0], content_list_path)

    return target_path
