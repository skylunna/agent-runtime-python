"""
将 rag.chunk.embedding 列从 vector(512) 迁移到 vector(1024)。
已有数据会被置为 NULL，需要重新跑 ingestion 重新向量化。

用法:
    PYTHONPATH=./src uv run python scripts/migrate_embedding_dim.py
"""

import psycopg
from cdss_rag.core.config import settings

OLD_DIM = 512
NEW_DIM = settings.embedding_dim  # 1024


def main() -> None:
    print(f"Migrating rag.chunk.embedding: vector({OLD_DIM}) -> vector({NEW_DIM})")
    print(f"DSN: {settings.pg_dsn}")

    with psycopg.connect(settings.pg_dsn) as conn:
        with conn.cursor() as cur:
            # 1. 检查列是否存在
            cur.execute("""
                SELECT column_name, udt_name, character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'rag'
                  AND table_name = 'chunk'
                  AND column_name = 'embedding'
            """)
            row = cur.fetchone()
            if row is None:
                print("ERROR: rag.chunk.embedding column not found")
                return
            print(f"Current column type: {row[1]}")

            # 2. 检查数据量
            cur.execute("SELECT COUNT(*) FROM rag.chunk WHERE embedding IS NOT NULL")
            count = cur.fetchone()[0]
            print(f"Rows with non-null embedding: {count}")

            if count > 0:
                print(f"WARNING: {count} rows will have embedding set to NULL.")
                print("You must re-run the ingestion pipeline afterwards.")
                confirm = input("Continue? [y/N] ")
                if confirm.lower() != "y":
                    print("Aborted.")
                    return

            # 3. 方案 A: 使用临时列（推荐，更安全）
            print("Step 1: Adding temporary column...")
            cur.execute(f"""
                ALTER TABLE rag.chunk 
                ADD COLUMN embedding_new vector({NEW_DIM})
            """)
            
            print("Step 2: Dropping old column...")
            cur.execute("""
                ALTER TABLE rag.chunk 
                DROP COLUMN embedding CASCADE
            """)
            
            print("Step 3: Renaming new column...")
            cur.execute("""
                ALTER TABLE rag.chunk 
                RENAME COLUMN embedding_new TO embedding
            """)
            
            # 4. 重建索引
            print("Step 4: Rebuilding index...")
            cur.execute("""
                CREATE INDEX idx_chunk_embedding ON rag.chunk 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            conn.commit()
            print(f"Done. rag.chunk.embedding is now vector({NEW_DIM})")

            if count > 0:
                print(f"Reminder: {count} rows need re-embedding via ingestion pipeline.")


if __name__ == "__main__":
    main()