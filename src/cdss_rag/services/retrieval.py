import logging
from ..schemas.retrieve import RetrieveRequest, RetrieveResponse, RetrievedChunk


logger = logging.getLogger(__name__)


class RetrievalService:
    """检索服务 """

    async def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        logger.info(f"[STUB] Retrieve: kb={req.kb_id}, query={req.query}")
        return RetrieveResponse(
            query=req.query,
            results=[
                RetrievedChunk(
                    chunk_id="stub-1",
                    content=f"[STUB] 这是对 '{req.query}' 的占位检索结果",
                    document="占位文档.pdf",
                    page=1,
                    score=0.95,
                )
            ],
        )
    

retrieval_service = RetrievalService()