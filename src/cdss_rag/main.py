import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .core.config import settings
from .core.logging_conf import setup_logging
from .core.db import db_pool
from .core.errors import (
    CDSSException,
    cdss_exception_handler,
    generic_exception_handler,
)
from .api import health, ingest, retrieve, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CDSS RAG service (env={settings.app_env})...")

    await db_pool.open(wait=True, timeout=10)
    logger.info("DB pool opened")

    yield

    logger.info("shutting down...")
    await db_pool.close()
    logger.info("DB pool closed")


app = FastAPI(
    title="CDSS Agent RAG - Execution Plane",
    version="0.1.0",
    lifespan=lifespan,
)

# 异常处理
app.add_exception_handler(CDSSException, cdss_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 路由
app.include_router(health.router)
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(retrieve.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")