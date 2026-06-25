from pydantic import BaseModel, Field
from typing import Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID,由 Java 侧生成")
    kb_id: str = Field(..., description="知识库 ID")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = True


class Citation(BaseModel):
    chunk_id: str
    document: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []