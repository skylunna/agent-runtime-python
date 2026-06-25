from fastapi import APIRouter

from ..schemas.common import ApiResponse
from ..schemas.retrieve import RetrieveRequest, RetrieveResponse
from ..services.retrieval import retrieval_service

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("", response_model=ApiResponse[RetrieveResponse])
async def retrieve(req: RetrieveRequest):
    result = await retrieval_service.retrieve(req)
    return ApiResponse(data=result)