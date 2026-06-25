from fastapi import APIRouter

from ..schemas.common import ApiResponse
from ..schemas.kb import CreateKBRequest, KBInfo
from ..services.kb import kb_service

router = APIRouter(prefix="/kb", tags=["knowledge_base"])


@router.post("", response_model=ApiResponse[KBInfo])
async def create_kb(req: CreateKBRequest):
    kb = await kb_service.create_kb(
        kb_id=req.kb_id,
        name=req.name,
        description=req.description,
        domain=req.domain,
    )
    return ApiResponse(data=KBInfo(**kb))


@router.get("/{kb_id}", response_model=ApiResponse[KBInfo])
async def get_kb(kb_id: str):
    kb = await kb_service.get_kb(kb_id)
    return ApiResponse(data=KBInfo(**kb))


@router.get("", response_model=ApiResponse[list[KBInfo]])
async def list_kbs():
    kbs = await kb_service.list_kbs()
    return ApiResponse(data=[KBInfo(**kb) for kb in kbs])