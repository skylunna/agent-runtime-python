import asyncio
import json
import logging
from typing import AsyncIterator, Any
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..core.config import settings
from ..core.errors import LLMError
from .prompts import CDSS_SYSTEM_PROMPT
from .tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)


# =============================================================
# 事件协议 (与 chat service 的 SSE 协议对齐)
# =============================================================

@dataclass
class AgentEvent:
    """Agent 产生的事件,会被 chat service 转成 SSE"""
    event: str           # tool_call / tool_result / token / done / error
    data: dict


# =============================================================
# Agent 配置
# =============================================================

@dataclass
class AgentConfig:
    max_iterations: int = 5          # 最大循环次数,防死循环
    tool_timeout_sec: int = 30       # 单个工具调用超时
    temperature: float = 0.1
    model: str = field(default_factory=lambda: settings.deepseek_model)


# =============================================================
# Agent 主循环
# =============================================================

class Agent:
    """
    单 Agent + ReAct 循环
    
    工作流程:
      while not done and iter < max:
          1. 调 LLM (非流式) 看是否要工具调用
          2. 若要工具: 执行工具,把结果追加到 messages
          3. 若不要工具: 进入"最终回答"阶段,流式输出
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    async def run(
        self,
        user_messages: list[dict],
        ctx: dict,
    ) -> AsyncIterator[AgentEvent]:
        """
        运行 Agent
        
        Args:
            user_messages: 对话历史 [{"role":"user"/"assistant", "content":"..."}]
            ctx: 工具执行上下文,如 {"kb_id": "kb-diabetes-v1"}
        
        Yields:
            AgentEvent: 流式事件
        """
        messages = [
            {"role": "system", "content": CDSS_SYSTEM_PROMPT},
            *user_messages,
        ]

        for iteration in range(self.config.max_iterations):
            logger.info(f"[Agent] Iteration {iteration + 1}/{self.config.max_iterations}")

            # === 第一步: 调 LLM (非流式,需要看完整 response 判断 tool_calls) ===
            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.exception("LLM call failed in decision phase")
                yield AgentEvent("error", {"message": f"LLM error: {e}"})
                return

            choice = resp.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason

            # === 第二步: 判断是否要调工具 ===
            if msg.tool_calls:
                # LLM 决定调工具
                # 把 assistant 消息 (含 tool_calls) 加入 messages
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                }
                messages.append(assistant_msg)

                # 并发执行所有工具调用
                tool_tasks = []
                for tc in msg.tool_calls:
                    yield AgentEvent("tool_call", {
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
                    tool_tasks.append(self._exec_tool_with_timeout(
                        tc.function.name,
                        tc.function.arguments,
                        ctx,
                    ))

                results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                # 把工具结果加到 messages,并产出事件
                for tc, result in zip(msg.tool_calls, results):
                    if isinstance(result, Exception):
                        tool_content = json.dumps(
                            {"status": "error", "error": str(result)},
                            ensure_ascii=False
                        )
                    else:
                        tool_content = result.to_llm_string()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

                    yield AgentEvent("tool_result", {
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": tool_content,
                    })

                # 继续下一轮循环
                continue

            # === 第三步: LLM 决定不调工具 → 进入最终回答 ===
            # 这里需要重新调 LLM 拿流式输出
            # (上面的 resp 已经包含了完整 answer,但为了流式,我们重新跑一次 stream)
            # 优化: 如果上面 resp 已经有 content 且不需要流式,可以直接用,这里为教学清晰起见统一走 stream

            # 把已生成的最终 answer 也加到 messages (虽然没用到,但保持一致性)
            # 然后流式输出 - 但需要去掉最后一个 assistant
            # 简化方案: 直接对当前 messages (不含最终 assistant) 再调一次 stream
            full_answer = ""
            try:
                stream = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=self.config.temperature,
                    stream=True,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_answer += delta.content
                        yield AgentEvent("token", {"content": delta.content})

            except Exception as e:
                logger.exception("LLM call failed in answer phase")
                yield AgentEvent("error", {"message": f"LLM error: {e}"})
                return

            # 结束
            yield AgentEvent("done", {
                "iterations": iteration + 1,
                "total_answer_len": len(full_answer),
                "finish_reason": finish_reason,
            })
            return

        # 循环上限耗尽
        logger.warning("Agent reached max iterations")
        yield AgentEvent("done", {
            "iterations": self.config.max_iterations,
            "warning": "max_iterations_reached",
        })

    async def _exec_tool_with_timeout(self, name: str, args_json: str, ctx: dict):
        """带超时的工具执行"""
        try:
            return await asyncio.wait_for(
                execute_tool(name, args_json, ctx),
                timeout=self.config.tool_timeout_sec,
            )
        except asyncio.TimeoutError:
            from .tools import ToolResult
            return ToolResult(False, None, f"Tool {name} timeout after {self.config.tool_timeout_sec}s")


# 模块级单例
agent = Agent()