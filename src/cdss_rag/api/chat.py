import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse


from ..schemas.chat import ChatRequest
from ..schemas.common import ApiResponse
from ..services.chat import chat_service


router = APIRouter(prefix="/chat", tags=["chat"])


def _sse_format(event: str, data: dict) -> str:
    """SSE 标准格式: event: xxx\ndata: {json}\n\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(req: ChatRequest):
    """
    流式： 返回 SSE 流 (Content-Type: text/event-stream)
    非流式: 返回 JSON {code, message, data}
    """
    if req.stream:
        async def event_generator():
            try:
                async for evt in chat_service.chat_stream(req):
                    yield _sse_format(evt["event", evt["data"]])
            except Exception as e:
                yield _sse_format("error", {"message": str(e)})

        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # 兼容 nginx,禁止缓冲
            },
        )

    result = await chat_service.chat(req)
    return ApiResponse(data=result)
