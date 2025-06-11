# /llm_api/providers/enhanced_ollama_v2.py
import logging
from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability, LLMProvider
from ..cogniquantum import CogniQuantumSystemV2, ComplexityRegime
from ..utils.helper_functions import get_model_family

logger = logging.getLogger(__name__)

class EnhancedOllamaProviderV2(EnhancedLLMProvider):
    """
    V2 Enhanced Ollama Provider
    """

    def __init__(self, standard_provider: LLMProvider):
        """
        コンストラクタ。
        standard_providerを受け取り、親クラスの__init__を呼び出す。
        """
        super().__init__(standard_provider)

    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        標準的な呼び出しを、内部の標準プロバイダーに委任する。（必須実装）
        """
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """
        V2の拡張機能を使用すべきか判定する。（必須実装）
        """
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized'
        ]

    async def enhanced_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        論文の知見に基づき、Ollamaに対する推論プロセスを最適化する。
        """
        try:
            mode = kwargs.get('mode', 'adaptive')
            logger.info(f"Ollama V2拡張呼び出し実行 (モード: {mode})")
            
            force_regime = self._determine_force_regime(mode)
            base_model_kwargs = self._get_optimized_params(kwargs.get('model'), mode, kwargs)

            cq_system = CogniQuantumSystemV2(self.standard_provider, base_model_kwargs)
            
            result = await cq_system.solve_problem(
                prompt,
                system_prompt=system_prompt,
                force_regime=force_regime
            )

            # CogniQuantumSystemからのエラーをチェックして伝達する
            if not result.get('success'):
                error_message = result.get('error', 'CogniQuantumシステムで不明なエラーが発生しました。')
                logger.error(f"CogniQuantumシステムがエラーを返しました: {error_message}")
                return {"text": "", "error": error_message}

            # 成功した場合、応答をフォーマットする
            return {
                'text': result.get('final_solution', ''),
                'model': base_model_kwargs.get('model', 'unknown'),
                'usage': {},
                'error': None, # 成功時はエラーなし
                'version': 'v2',
                'paper_based_improvements': result.get('complexity_analysis')
            }

        except Exception as e:
            # この拡張プロバイダー自身のロジックでエラーが発生した場合
            logger.error(f"Ollama V2拡張プロバイダーで予期せぬエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}

    def _determine_force_regime(self, mode: str) -> ComplexityRegime | None:
        """モードから強制する複雑性レジームを決定する"""
        if mode == 'efficient':
            return ComplexityRegime.LOW
        if mode == 'balanced':
            return ComplexityRegime.MEDIUM
        if mode == 'decomposed':
            return ComplexityRegime.HIGH
        return None # adaptive または paper_optimized

    def _get_optimized_params(self, model_name: str, mode: str, kwargs: Dict) -> Dict:
        """
        Ollamaのモデルファミリーとモードに基づいてパラメータを最適化する。
        """
        params = kwargs.copy()
        
        # --modelが指定されていない場合、デフォルトの'gemma3:latest'を使用
        effective_model_name = model_name or 'gemma3:latest'
        
        family = get_model_family(effective_model_name)
        logger.info(f"モデル '{effective_model_name}' (ファミリー: {family}) のパラメータを最適化")

        # デフォルトモデル設定
        if not params.get('model'):
            params['model'] = effective_model_name
        
        # 温度設定
        temp_map = {
            'efficient': 0.2,
            'balanced': 0.5,
            'decomposed': 0.4,
        }
        if mode in temp_map and 'temperature' not in params:
            params['temperature'] = temp_map[mode]

        # Llamaファミリー向け最適化
        if family == 'llama':
            if 'top_p' not in params:
                params['top_p'] = 0.9
        
        # Qwenファミリー向け最適化
        elif family == 'qwen':
             if 'temperature' not in params:
                params['temperature'] = temp_map.get(mode, 0.4)
        
        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: True,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: True,
        }