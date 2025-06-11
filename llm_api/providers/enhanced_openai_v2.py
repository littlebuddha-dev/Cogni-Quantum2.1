# /llm_api/providers/enhanced_openai_v2.py
import logging
from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability, LLMProvider
from ..cogniquantum import CogniQuantumSystemV2, ComplexityRegime

logger = logging.getLogger(__name__)

class EnhancedOpenAIProviderV2(EnhancedLLMProvider):
    """V2 Enhanced OpenAI Provider"""

    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized'
        ]
    
    async def enhanced_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        try:
            mode = kwargs.get('mode', 'adaptive')
            logger.info(f"OpenAI V2拡張呼び出し実行 (モード: {mode})")
            
            force_regime = self._determine_force_regime(mode)
            base_model_kwargs = self._get_optimized_params(mode, kwargs)
            
            cq_system = CogniQuantumSystemV2(self.standard_provider, base_model_kwargs)
            
            result = await cq_system.solve_problem(
                prompt,
                system_prompt=system_prompt,
                force_regime=force_regime
            )
            
            if not result.get('success'):
                error_message = result.get('error', 'CogniQuantumシステム(OpenAI)で不明なエラーが発生しました。')
                logger.error(f"CogniQuantumシステムがエラーを返しました: {error_message}")
                return {"text": "", "error": error_message}
            
            return {
                'text': result.get('final_solution', ''),
                'model': base_model_kwargs.get('model', 'gpt-4o'),
                'usage': {},
                'error': None,
                'version': 'v2',
                'paper_based_improvements': result.get('complexity_analysis')
            }
        except Exception as e:
            logger.error(f"OpenAI V2拡張プロバイダーでエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}

    def _determine_force_regime(self, mode: str) -> ComplexityRegime | None:
        if mode == 'efficient': return ComplexityRegime.LOW
        if mode == 'balanced': return ComplexityRegime.MEDIUM
        if mode == 'decomposed': return ComplexityRegime.HIGH
        return None

    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        params = kwargs.copy()
        if 'model' not in params:
            params['model'] = 'gpt-4o'
        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: True,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: True,
            ProviderCapability.JSON_MODE: True,
        }