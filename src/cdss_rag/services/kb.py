import logging
from ..core.db import db_pool
from ..core.errors import NotFoundError, ValidationError
from ..llm.embedding import embedding_service
from ..repositories.kb_repo import KnowledgeBaseRepo

logger = logging.getLogger(__name__)


class KBService:

    async def create_kb(self, *, kb_id: str, name: str,
                        description: str | None = None,
                        domain: str = "medical") -> dict:
        # 防止重复创建
        async with db_pool.connection() as conn:
            existing = await KnowledgeBaseRepo.get(conn, kb_id)
            if existing:
                raise ValidationError(f"KB already exists: {kb_id}")

            # embedding 模型信息从当前服务取,锁定到 kb 元数据
            kb = await KnowledgeBaseRepo.create(
                conn,
                kb_id=kb_id,
                name=name,
                description=description,
                domain=domain,
                embedding_model=embedding_service.model_name,
                embedding_dim=embedding_service.dim,
            )
            await conn.commit()
            logger.info(f"KB created: {kb_id}")
            return kb

    async def get_kb(self, kb_id: str) -> dict:
        async with db_pool.connection() as conn:
            kb = await KnowledgeBaseRepo.get(conn, kb_id)
            if not kb:
                raise NotFoundError(f"KB not found: {kb_id}")
            return kb

    async def list_kbs(self) -> list[dict]:
        async with db_pool.connection() as conn:
            return await KnowledgeBaseRepo.list_all(conn)


kb_service = KBService()