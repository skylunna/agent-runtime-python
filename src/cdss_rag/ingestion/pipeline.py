import logging
import uuid
from dataclasses import dataclass


from .loader import load_pdf, LoadedDocument
from .splitter import split_document, SplittedChunk
from ..llm.embedding import embedding_service


logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    document: LoadedDocument
    chunks: list[SplittedChunk]
    embeddings: list[list[float]]   # 每个 chunk 对应一个向量


def run_ingestion(file_path: str,
                  chunk_size: int = 500,
                  chunk_overlap: int = 50) -> IngestionResult:
    """
    摄入 pipeline （无DB操作）
    单一职责：文件 - 切片 + 向量
    DB 写入交给 service 层处理
    """
    document_id = f"doc-{uuid.uuid4().hex[:8]}"

    # 1. 加载
    doc = load_pdf(file_path)

    # 2. 切片
    chunks = split_document(doc, chunk_size, chunk_overlap)

    if not chunks:
        logger.warning(f"No chunks generated from {file_path}")
        return IngestionResult(
            document_id=document_id,
            document=doc,
            chunks=[],
            embeddings=[],
        )
    
    # 3. 向量化 (批量 效率最优)
    logger.info(f"Encoding {len(chunks)} chunks...")
    texts = [c.content for c in chunks]
    vectors = embedding_service.encode(texts)
    embeddings = vectors.tolist()   # numpy -> list[list[float]]

    return IngestionResult(
        document_id=document_id,
        document=doc,
        chunks=chunks,
        embeddings=embeddings,
    )