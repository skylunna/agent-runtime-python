import asyncio
import logging

from ..core.db import db_pool
from psycopg.types.json import Jsonb
from ..core.errors import NotFoundError, ValidationError
from ..ingestion.pipeline import run_ingestion
from ..repositories.kb_repo import KnowledgeBaseRepo
from ..repositories.document_repo import DocumentRepo
from ..repositories.chunk_repo import ChunkRepo
from ..schemas.ingest import IngestRequest, IngestResponse


logger = logging.getLogger(__name__)


class IngestionService:

    async def ingest(self, req: IngestRequest) -> IngestRequest:
        # 1. 校验KB存在
        async with db_pool.connection() as conn:
            kb = await KnowledgeBaseRepo.get(conn, req.kb_id)
            if not kb:
                raise NotFoundError(f"KB not found: {req.kb_id}")
            

        # 2. 跑 pipeline (cpu密集，放到线程池)
        # asyncio.to_thread 是 Python 3.9+ 的简化写法
        logger.info(f"Starting ingestion: kb={req.kb_id}, file={req.file_path}")
        try:
            result = await asyncio.to_thread(run_ingestion, req.file_path)
        except FileNotFoundError as e:
            raise ValidationError(str(e))
        
        if not result.chunks:
            raise ValidationError("No content extracted from the document")
        

        # 3. 写库 (单事务)
        async with db_pool.connection() as conn:
            # 文档去重检查
            existing = await DocumentRepo.find_by_hash(
                conn, req.kb_id, result.document.file_hash
            )
            if existing:
                logger.warning(f"Duplicate document detected: {existing['id']}")
                raise ValidationError(
                    f"Document already exists with same hash: {existing['id']}"
                )

            # 插入文档
            await DocumentRepo.insert(
                conn,
                doc_id=result.document_id,
                kb_id=req.kb_id,
                title=result.document.title,
                source_path=result.document.source_path,
                file_hash=result.document.file_hash,
                page_count=result.document.page_count,
                metadata={},
                status="ready",
            )

            # 准备 chunk 批量插入数据
            chunk_rows = []
            for chunk, vector in zip(result.chunks, result.embeddings):
                chunk_rows.append({
                    "id": chunk.id,
                    "kb_id": req.kb_id,
                    "document_id": result.document_id,
                    "seq": chunk.seq,
                    "content": chunk.content,
                    "content_len": len(chunk.content),
                    "page": chunk.page,
                    "metadata": Jsonb({}),
                    "embedding": vector,
                })

            inserted = await ChunkRepo.bulk_insert(conn, chunk_rows)
            await conn.commit()

            logger.info(f"Ingestion done: doc={result.document_id}, "
                        f"chunks={inserted}")

        return IngestResponse(
            kb_id=req.kb_id,
            document_id=result.document_id,
            chunks=inserted,
            status="ready",
        )
    
    
ingestion_service = IngestionService()