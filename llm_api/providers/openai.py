# /llm_api/providers/openai.py
import logging
import os
from typing import Any, Dict

from openai import AsyncOpenAI
from .base import LLMProvider, ProviderCapability

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """
    OpenAI APIと対話するための標準プロバイダー
    """
    def __init__(self, model: str = None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or "gpt-4o"
        super().__init__()

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: True,
            ProviderCapability.JSON_MODE: True,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False
        
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """OpenAI APIを呼び出し、標準化された辞書形式で結果を返す。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            
            content = response.choices[0].message.content
            usage = response.usage

            return {
                "text": content.strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "error": None,
            }
        except Exception as e:
            logger.error(f"OpenAI API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}