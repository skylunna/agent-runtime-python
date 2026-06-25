import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader


logger = logging.getLogger(__name__)


@dataclass
class LoadedPage:
    """一个文档加载后的页面单元"""
    page_no: int    # 1-based
    content: str

@dataclass
class LoadedDocument:
    """加载完成的文档"""
    title: str
    source_path: str
    file_hash: str
    page_count: int
    pages: list[LoadedPage]


def load_pdf(file_path: str) -> LoadedDocument:
    """
    加载PDF
    MVP阶段：用pypdf，简单可靠
    Phase 2：升级到 pdfplumber/MinerU 处理表格
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF supported in MVP, got: {path.suffix}")
    

    # 计算文件 hash （去重用）
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()


    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(LoadedPage(page_no=i, content=text))

    logger.info(f"Loaded PDF: {path.name}, total_pages={len(reader.pages)}, "
            f"non_empty_pages={len(pages)}")
    
    return LoadedDocument(
        title=path.stem,
        source_path=str(path.absolute()),
        file_hash=file_hash,
        page_count=len(reader.pages),
        pages=pages,
    )