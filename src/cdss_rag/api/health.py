from fastapi import APIRouter
from ..core.db import db_health_check


router = APIRouter(tags=["health"])


@router.get("/healthz", summary="存活探针")
async def healthz():
    """k8s liveness probe"""
    return {"status": "ok"}


@router.get("/readyz", summary="就绪探针")
async def readyz():
    """k8s readiness probe - 检查依赖资源"""
    db_ok = await db_health_check()
    if db_ok:
        return {"status": "ready", "checks": {"db": "ok"}}
    return {"status": "not_ready", "checks": {"db": "fail"}}