# /llm_api/providers/gemini.py
import logging
import os
from typing import Any, Dict

import google.generativeai as genai
from .base import LLMProvider, ProviderCapability

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """
    Google Gemini APIと対話するための標準プロバイダー
    """
    def __init__(self, model: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY環境変数が設定されていません。")
        genai.configure(api_key=api_key)
        self.model_name = model or "gemini-1.5-flash"
        self.model = genai.GenerativeModel(self.model_name)
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
        """Gemini APIを呼び出し、標準化された辞書形式で結果を返す。"""
        try:
            # system_promptはgeneration_configの一部として渡すのが一般的
            config = {}
            if system_prompt:
                # Geminiではsystem_instructionとして渡す
                # ただし、モデルによってはサポートが異なる
                # ここでは簡易的にプロンプトに含める
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            response = await self.model.generate_content_async(full_prompt)
            
            return {
                "text": response.text.strip(),
                "model": self.model_name,
                "usage": {}, # Gemini APIは現在、トークン使用量を直接返さない
                "error": None,
            }
        except Exception as e:
            logger.error(f"Gemini API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}