import logging
from typing import AsyncIterator

from ..agent.loop import agent
from ..schemas.chat import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)


class ChatService:

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict]:
        """
        流式 chat - 接入 Agent
        协议见 Step 2 SSE 设计 (这里扩展了 tool_call / tool_result)
        """
        # 把 ChatMessage 转成 OpenAI 格式
        user_messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages
        ]

        # Agent 执行上下文
        ctx = {
            "kb_id": req.kb_id,
            "session_id": req.session_id,
        }

        logger.info(f"[Chat] session={req.session_id}, kb={req.kb_id}, "
                    f"messages={len(user_messages)}")

        async for event in agent.run(user_messages, ctx):
            # AgentEvent -> SSE dict
            yield {"event": event.event, "data": event.data}

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """非流式: 收集所有 token 拼起来"""
        full_answer = ""
        citations: list[Citation] = []

        user_messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages
        ]
        ctx = {"kb_id": req.kb_id, "session_id": req.session_id}

        async for event in agent.run(user_messages, ctx):
            if event.event == "token":
                full_answer += event.data.get("content", "")
            elif event.event == "tool_result":
                # 从检索结果里提取 citations
                tool_name = event.data.get("name")
                if tool_name == "search_knowledge_base":
                    try:
                        import json
                        content = json.loads(event.data.get("content", "{}"))
                        if content.get("status") == "ok":
                            for chunk in content.get("data", {}).get("chunks", []):
                                citations.append(Citation(
                                    chunk_id=chunk["chunk_id"],
                                    document=chunk["document"],
                                    page=chunk.get("page"),
                                    score=chunk.get("score", 0.0),
                                ))
                    except Exception:
                        pass

        return ChatResponse(
            session_id=req.session_id,
            answer=full_answer,
            citations=citations,
        )


chat_service = ChatService()