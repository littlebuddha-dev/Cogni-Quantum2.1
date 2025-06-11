# /llm_api/providers/base.py
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ProviderCapability(Enum):
    """プロバイダーの機能を定義するEnum"""
    STANDARD_CALL = "standard_call"
    ENHANCED_CALL = "enhanced_call"
    STREAMING = "streaming"
    SYSTEM_PROMPT = "system_prompt"
    TOOLS = "tools"
    JSON_MODE = "json_mode"

class LLMProvider(ABC):
    """
    全てのLLMプロバイダーの抽象基底クラス（ABC）
    """
    def __init__(self):
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self.capabilities = self._get_default_capabilities()

    def _get_default_capabilities(self) -> Dict[ProviderCapability, bool]:
        """capabilitiesのデフォルト値を生成する。実装クラスでオーバーライド推奨。"""
        # get_capabilitiesが実装されていない場合に備えたフォールバック
        if hasattr(self, 'get_capabilities'):
            try:
                # get_capabilitiesがabstract methodではなくなった場合に備える
                return self.get_capabilities()
            except TypeError: # abstract methodのままの場合
                 pass
        return {cap: False for cap in ProviderCapability}


    @abstractmethod
    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """プロバイダーの機能を定義した辞書を返す。"""
        pass

    async def call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        メインの呼び出しメソッド。
        拡張機能を使うべきか判断し、enhanced_callまたはstandard_callに処理を振り分ける。
        """
        # 拡張機能が利用可能 かつ 拡張機能を使用すべき状況か を判断
        use_enhancement = self.get_capabilities().get(ProviderCapability.ENHANCED_CALL, False) and \
                          self.should_use_enhancement(prompt, **kwargs)

        if use_enhancement:
            # hasattrでenhanced_callメソッドの存在を安全に確認
            if hasattr(self, 'enhanced_call') and callable(self.enhanced_call):
                logger.debug(f"プロバイダー '{self.provider_name}' の enhanced_call を呼び出します。")
                # mypyに型を認識させるため getattr を使用
                enhanced_call_method = getattr(self, "enhanced_call")
                return await enhanced_call_method(prompt, system_prompt, **kwargs)
            else:
                 logger.warning(f"'{self.provider_name}' はENHANCED_CALLケイパビリティを持つと報告しましたが、enhanced_callメソッドが見つかりません。")

        # デフォルトでstandard_callを呼び出す
        logger.debug(f"プロバイダー '{self.provider_name}' の standard_call を呼び出します。")
        return await self.standard_call(prompt, system_prompt, **kwargs)

    @abstractmethod
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        標準的な（拡張機能なしの）LLM呼び出し。全ての具象プロバイダーで実装必須。
        """
        pass

    @abstractmethod
    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """
        拡張機能（enhanced_call）を使用すべきかどうかを判断する。
        全ての具象プロバイダーで実装必須。
        """
        pass

class EnhancedLLMProvider(LLMProvider):
    """
    標準プロバイダーをラップし、追加機能を提供する拡張プロバイダーの基底クラス。
    """
    def __init__(self, standard_provider: LLMProvider):
        super().__init__()
        self.standard_provider = standard_provider
        # provider_nameはラップしているプロバイダーのものを引き継ぐ
        self.provider_name = standard_provider.provider_name

    @abstractmethod
    async def enhanced_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        拡張された推論ロジック。全ての拡張プロバイダーで実装必須。
        """
        pass