from pydantic import BaseModel, Field
from datetime import datetime


class CreateKBRequest(BaseModel):
    kb_id: str = Field(..., pattern=r"^[a-z0-9\-]+$", max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = None
    domain: str = Field(default="medical")


class KBInfo(BaseModel):
    id: str
    name: str
    description: str | None
    domain: str
    embedding_model: str
    embedding_dim: int
    created_at: datetime
    updated_at: datetime