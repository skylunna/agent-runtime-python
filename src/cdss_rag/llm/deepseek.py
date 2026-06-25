import logging
from typing import AsyncIterator
from openai import AsyncOpenAI


from ..core.config import settings
from ..core.errors import LLMError


logger = logging.getLogger(__name__)


class DeepSeekClient:
    """DeepSeek API 封装"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self.model = settings.deepseek_model

    async def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.exception("DeepSeek chat failed")
            raise LLMError(f"DeepSeek chat failed: {e}") from e
        
    async def chat_stream(
            self, messages: list[dict], temperature: float = 0.1
    ) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.exception("DeepSeek stream failed")
            raise LLMError(f"DeepSeek stream failed: {e}") from e
        

# 单例
deepseek_client = DeepSeekClient()