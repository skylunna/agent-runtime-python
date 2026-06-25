import asyncio
import logging
from typing import AsyncIterator
from ..schemas.chat import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)


class ChatService:
    """对话服务 """

    async def chat(self, req: ChatRequest) -> ChatResponse:
        logger.info(f"[STUB] Chat (non-stream): session={req.session_id}")
        user_msg = req.messages[-1].content
        return ChatResponse(
            session_id=req.session_id,
            answer=f"[STUB 非流式回答] 你问的是: {user_msg}",
            citations=[],
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict]:
        """
        流式输出协议(SSE event 格式):
          event=citation: 检索到的引用,在 token 流之前推送
          event=token:    模型 token,增量推送
          event=done:     结束信号,包含最终元数据
          event=error:    错误信号
        """
        logger.info(f"[STUB] Chat (stream): session={req.session_id}")
        user_msg = req.messages[-1].content

        # 1. 先吐 citation
        yield {
            "event": "citation",
            "data": {
                "chunk_id": "stub-1",
                "document": "占位文档.pdf",
                "page": 1,
                "score": 0.95,
            },
        }

        # 2. 模拟 token 流
        fake_answer = f"[STUB 流式] 收到你的问题「{user_msg}」。Step 4 会接入真实的 Agent + LLM。"
        for ch in fake_answer:
            yield {"event": "token", "data": {"content": ch}}
            await asyncio.sleep(0.02)

        # 3. 结束信号
        yield {
            "event": "done",
            "data": {"session_id": req.session_id, "total_tokens": len(fake_answer)},
        }


chat_service = ChatService()