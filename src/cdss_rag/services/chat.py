import logging
from typing import AsyncIterator

from ..agent.loop import agent
from ..schemas.chat import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)


class ChatService:

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict]:
        user_messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages
        ]
        ctx = {
            "kb_id": req.kb_id,
            "session_id": req.session_id,
        }

        # 用 Java 传来的配置;若没传,降级到 Agent 内置默认
        cfg = req.agent_config
        if cfg:
            logger.info(f"[Chat] session={req.session_id}, agent={cfg.agent_id}, "
                        f"messages={len(user_messages)}")
            event_iter = agent.run_with_config(
                user_messages=user_messages,
                ctx=ctx,
                system_prompt=cfg.system_prompt,
                model=cfg.model,
                temperature=cfg.temperature,
                max_iterations=cfg.max_iterations,
                tools_filter=cfg.tools_enabled,
            )
        else:
            # 向后兼容: 没传 agent_config 走默认
            logger.info(f"[Chat] session={req.session_id}, no agent_config (legacy)")
            event_iter = agent.run(user_messages, ctx)

        async for event in event_iter:
            yield {"event": event.event, "data": event.data}

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """非流式 - 同 Step 4"""
        full_answer = ""
        citations: list[Citation] = []
        async for event in self.chat_stream(req):
            if event["event"] == "token":
                full_answer += event["data"].get("content", "")
            elif event["event"] == "tool_result":
                tool_name = event["data"].get("name")
                if tool_name == "search_knowledge_base":
                    try:
                        import json
                        content = json.loads(event["data"].get("content", "{}"))
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