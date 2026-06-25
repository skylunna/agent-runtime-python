from typing import Optional
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class DocumentRepo:

    @staticmethod
    async def insert(conn: AsyncConnection, *,
                     doc_id: str, kb_id: str,
                     title: str, source_path: str,
                     file_hash: str, page_count: int,
                     metadata: dict | None = None,
                     status: str = "ready") -> None:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO rag.document
                    (id, kb_id, title, source_path, file_hash,
                     page_count, metadata, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (doc_id, kb_id, title, source_path, file_hash,
                 page_count, Jsonb(metadata or {}), status),
            )

    @staticmethod
    async def find_by_hash(conn: AsyncConnection,
                           kb_id: str, file_hash: str) -> Optional[dict]:
        """用于去重: 同一 kb 内同 hash 的文档"""
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, title FROM rag.document
                WHERE kb_id = %s AND file_hash = %s AND deleted_at IS NULL
                """,
                (kb_id, file_hash),
            )
            return await cur.fetchone()