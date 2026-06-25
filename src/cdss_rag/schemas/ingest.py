from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    kb_id: str = Field(..., description="知识库ID")
    file_path: str = Field(..., description="服务器本地文件路径 (MVP 简化方案)")


class IngestResponse(BaseModel):
    kb_id: str
    document_id: str
    chunks: int
    status: str