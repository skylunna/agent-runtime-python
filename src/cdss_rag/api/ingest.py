from fastapi import APIRouter

from ..schemas.common import ApiResponse
from ..schemas.ingest import IngestRequest, IngestResponse
from ..services.ingestion import ingestion_service

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=ApiResponse[IngestResponse])
async def ingest(req: IngestRequest):
    result = await ingestion_service.ingest(req)
    return ApiResponse(data=result)