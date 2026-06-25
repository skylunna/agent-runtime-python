import logging
import uuid
from ..schemas.ingest import IngestRequest, IngestResponse


logger = logging.getLogger(__name__)


class IngestionService:
    """文档摄入服务 - Step 3 会接入真实的 PDF 解析和向量化"""

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        logger.info(f"[STUB] ingest request: kb_id={req.kb_id}, file={req.file_path}")
        return IngestResponse(
            kb_id=req.kb_id,
            document_id=f"doc-{uuid.uuid4().hex[:8]}",
            chunks=0,
            status="stub_ok",
        )
    

ingestion_service = IngestionService()