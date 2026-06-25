import logging
from ..core.db import db_pool
from ..core.errors import NotFoundError
from ..llm.embedding import embedding_service
from ..repositories.kb_repo import KnowledgeBaseRepo
from ..repositories.chunk_repo import ChunkRepo
from ..schemas.retrieve import RetrieveRequest, RetrieveResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalService:

    async def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        # 1. 校验 KB
        async with db_pool.connection() as conn:
            kb = await KnowledgeBaseRepo.get(conn, req.kb_id)
            if not kb:
                raise NotFoundError(f"KB not found: {req.kb_id}")

        # 2. Query 向量化 (CPU 密集,丢线程池)
        import asyncio
        query_vec = await asyncio.to_thread(
            embedding_service.encode_one, req.query
        )
        query_vec_list = query_vec.tolist()

        # 3. 检索
        async with db_pool.connection() as conn:
            rows = await ChunkRepo.search(
                conn,
                kb_id=req.kb_id,
                query_embedding=query_vec_list,
                top_k=req.top_k,
            )

        results = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                content=r["content"],
                document=r["document_title"],
                page=r["page"],
                score=float(r["score"]),
            )
            for r in rows
        ]

        logger.info(f"Retrieved {len(results)} chunks for query: '{req.query[:50]}'")
        return RetrieveResponse(query=req.query, results=results)


retrieval_service = RetrievalService()