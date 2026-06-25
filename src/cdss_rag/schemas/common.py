from pydantic import BaseModel
from typing import Generic, TypeVar


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应包装，便于 Java 侧解析"""
    code: str = "OK"
    message: str = "success"
    data: T | None = None