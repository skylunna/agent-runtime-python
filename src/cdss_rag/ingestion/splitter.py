import logging
import uuid
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


from .loader import LoadedDocument


logger = logging.getLogger(__name__)


@dataclass
class SplittedChunk:
    """一个切片"""
    id: str
    seq: int        # 在文档内的顺序
    content: str
    page: int | None    # 来源页码


def split_document(doc: LoadedDocument,
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> list[SplittedChunk]:
    """
    切片
    mvp: 固定长度 + overlap
    Phase 2: 父子切片 + 结构感知切片
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,
    )

    chunks: list[SplittedChunk] = []
    seq = 0
    for page in doc.pages:
        page_chunks = splitter.split_text(page.content)
        for c in page_chunks:
            c_stripped = c.strip()
            if not c_stripped:
                continue
            chunks.append(SplittedChunk(
                id=f"chk-{uuid.uuid4().hex[:12]}",
                seq=seq,
                content=c_stripped,
                page=page.page_no,
            ))
            seq += 1

    logger.info(f"Split into {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks