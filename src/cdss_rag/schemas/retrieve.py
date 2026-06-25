from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    kb_id: str = Field(..., description="知识库 ID")
    query: str = Field(..., min_length=1, description="检索 query")
    top_k: int = Field(default=4, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    document: str
    page: int | None = None
    score: float


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]