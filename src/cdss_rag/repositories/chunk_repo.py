from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class ChunkRepo:

    @staticmethod
    async def bulk_insert(conn: AsyncConnection, rows: list[dict]) -> int:
        """
        批量插入 chunks
        每行 dict 需包含: id, kb_id, document_id, seq, content, content_len,
                       page, metadata, embedding (list[float])
        """
        if not rows:
            return 0

        async with conn.cursor() as cur:
            # psycopg 3 的 executemany 配合 RETURNING 不友好,这里用 copy 或循环
            # MVP 阶段用 executemany 即可,性能足够
            await cur.executemany(
                """
                INSERT INTO rag.chunk
                    (id, kb_id, document_id, seq, content, content_len,
                     page, metadata, embedding)
                VALUES (%(id)s, %(kb_id)s, %(document_id)s, %(seq)s,
                        %(content)s, %(content_len)s, %(page)s,
                        %(metadata)s, %(embedding)s)
                """,
                rows,
            )
            return len(rows)

    @staticmethod
    async def search(conn: AsyncConnection, *,
                     kb_id: str,
                     query_embedding: list[float],
                     top_k: int = 4) -> list[dict]:
        """
        向量检索
        距离算子:
          <-> L2 距离
          <#> 内积
          <=> 余弦距离 (与归一化 embedding + vector_cosine_ops 配套)
        """
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT
                    c.id              AS chunk_id,
                    c.content         AS content,
                    c.page            AS page,
                    c.document_id     AS document_id,
                    d.title           AS document_title,
                    (1 - (c.embedding <=> %s::vector)) AS score
                FROM rag.chunk c
                JOIN rag.document d ON d.id = c.document_id
                WHERE c.kb_id = %s
                  AND c.deleted_at IS NULL
                  AND d.deleted_at IS NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, kb_id, query_embedding, top_k),
            )
            return await cur.fetchall()