from typing import Optional
from psycopg import AsyncConnection
from psycopg.rows import dict_row


class KnowledgeBaseRepo:
    """知识库 Repository"""

    @staticmethod
    async def get(conn: AsyncConnection, kb_id: str) -> Optional[dict]:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, name, description, domain,
                       embedding_model, embedding_dim,
                       created_at, updated_at
                FROM rag.knowledge_base
                WHERE id = %s AND deleted_at IS NULL
                """,
                (kb_id,),
            )
            return await cur.fetchone()
        

    @staticmethod
    async def list_all(conn: AsyncConnection) -> list[dict]:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, name, description, domain,
                       embedding_model, embedding_dim,
                       created_at, updated_at
                FROM rag.knowledge_base
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                """
            )
            return await cur.fetchall()
        

    @staticmethod
    async def create(conn: AsyncConnection, *,
                     kb_id: str, name: str,
                     description: str | None,
                     domain: str,
                     embedding_model: str,
                     embedding_dim: int) -> dict:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO rag.knowledge_base
                    (id, name, description, domain, embedding_model, embedding_dim)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, name, description, domain,
                          embedding_model, embedding_dim,
                          created_at, updated_at
                """,
                (kb_id, name, description, domain, embedding_model, embedding_dim),
            )
            return await cur.fetchone()