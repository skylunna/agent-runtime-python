from pydantic import BaseModel, Field
from typing import Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AgentConfig(BaseModel):
    """从 Java 管理面传过来的 Agent配置"""
    agent_id: str
    system_prompt: str
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_iterations: int = 5
    tools_enabled: list[str] = Field(default_factory=list)

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID,由 Java 侧生成")
    kb_id: str = Field(..., description="知识库 ID")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = True
    agent_config: AgentConfig | None = None


class Citation(BaseModel):
    chunk_id: str
    document: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []