# /llm_api/providers/huggingface.py
import logging
import os
from typing import Any, Dict

from huggingface_hub import AsyncInferenceClient
from .base import LLMProvider, ProviderCapability

logger = logging.getLogger(__name__)

class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face Inference APIと対話するための標準プロバイダー
    """
    def __init__(self, model: str = None):
        self.client = AsyncInferenceClient(token=os.getenv("HF_TOKEN"))
        self.model = model or "meta-llama/Meta-Llama-3-8B-Instruct"
        super().__init__()

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: False,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: False,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False
        
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Hugging Face Inference APIを呼び出し、標準化された辞書形式で結果を返す。"""
        # system_promptをHuggingFaceの形式に合わせる
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        else:
            full_prompt = prompt
            
        try:
            response_text = await self.client.text_generation(
                prompt=full_prompt,
                model=kwargs.get("model", self.model),
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            return {
                "text": response_text.strip(),
                "model": kwargs.get("model", self.model),
                "usage": {}, # HF APIはトークン使用量を返さない
                "error": None,
            }
        except Exception as e:
            logger.error(f"Hugging Face API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}