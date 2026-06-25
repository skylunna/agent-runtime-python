import logging
from psycopg_pool import AsyncConnectionPool


from .config import settings


logger = logging.getLogger(__name__)


# 全局连接池，生命周期由 main.py 的 lifespan 管理
db_pool = AsyncConnectionPool(
    conninfo=settings.pg_dsn,
    min_size=settings.pg_pool_min,
    max_size=settings.pg_pool_max,
    open=False
)


async def db_health_check() -> bool:
    """探针：验证DB连通性"""
    try:
        async with db_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()
            return row[0] == 1
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False