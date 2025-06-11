# /llm_api/cogniquantum/engine.py
"""
論文の発見に基づく改良推論エンジン
"""
import logging
from typing import Any, Dict, List

from .analyzer import AdaptiveComplexityAnalyzer
from .enums import ComplexityRegime
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

class EnhancedReasoningEngine:
    """論文の発見に基づく改良推論エンジン"""
    
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any]):
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        self.complexity_analyzer = AdaptiveComplexityAnalyzer()
        
    async def execute_reasoning(
        self,
        prompt: str,
        system_prompt: str = "",
        complexity_score: float = None,
        regime: ComplexityRegime = None
    ) -> Dict[str, Any]:
        """複雑性体制に応じた適応的推論の実行"""
        
        if complexity_score is None or regime is None:
            complexity_score, regime = self.complexity_analyzer.analyze_complexity(prompt)
        
        logger.info(f"推論実行開始: 体制={regime.value}, 複雑性={complexity_score:.2f}")
        
        if regime == ComplexityRegime.LOW:
            return await self._execute_low_complexity_reasoning(prompt, system_prompt)
        elif regime == ComplexityRegime.MEDIUM:
            return await self._execute_medium_complexity_reasoning(prompt, system_prompt)
        else:  # HIGH
            return await self._execute_high_complexity_reasoning(prompt, system_prompt)
    
    async def _execute_low_complexity_reasoning(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """低複雑性問題の推論（overthinking防止）"""
        logger.info("低複雑性推論モード: 簡潔・効率重視")
        
        efficient_prompt = f"""以下の問題に対して、簡潔で効率的な解答を提供してください。
過度な分析や長時間の検討は避け、直接的なアプローチを取ってください。

問題: {prompt}

重要: 最初に思いついた合理的な解答が往々にして正解です。"""
        
        response = await self.provider.call(efficient_prompt, system_prompt, **self.base_model_kwargs)
        
        return {
            'solution': response.get('text', ''),
            'error': response.get('error'), # エラー情報を伝達
            'complexity_regime': ComplexityRegime.LOW.value,
            'reasoning_approach': 'efficient_direct',
            'overthinking_prevention': True
        }
    
    async def _execute_medium_complexity_reasoning(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """中程度複雑性問題の推論（最適体制）"""
        logger.info("中複雑性推論モード: バランス型思考")
        
        structured_prompt = f"""以下の中程度の複雑性を持つ問題を、段階的かつ体系的に解決してください。

問題: {prompt}

推論プロセス:
1. 問題の核心的要素を特定
2. 解決に必要な情報を整理
3. 段階的解決戦略を構築
4. 各段階を実行し、中間結果を検証
5. 最終解を統合

各段階で中間結果を明示し、次の段階への論理的接続を示してください。"""
        
        response = await self.provider.call(structured_prompt, system_prompt, **self.base_model_kwargs)
        
        return {
            'solution': response.get('text', ''),
            'error': response.get('error'), # エラー情報を伝達
            'complexity_regime': ComplexityRegime.MEDIUM.value,
            'reasoning_approach': 'structured_progressive',
            'stage_verification': True
        }
    
    async def _execute_high_complexity_reasoning(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """高複雑性問題の推論（崩壊回避戦略）"""
        logger.info("高複雑性推論モード: 分解・段階実行")
        
        decomposition_result = await self._decompose_complex_problem(prompt, system_prompt)
        if decomposition_result.get('error'): # 分解でエラーが発生した場合
            return {'solution': '', 'error': decomposition_result['error']}

        staged_solutions = await self._solve_decomposed_problems(
            decomposition_result, prompt, system_prompt
        )
        if staged_solutions[-1].get('error'): # 段階的解決でエラーが発生した場合
             return {'solution': '', 'error': staged_solutions[-1]['error']}

        final_solution = await self._integrate_staged_solutions(
            staged_solutions, prompt, system_prompt
        )
        # 最終的なソリューションにエラーが含まれているか確認
        if isinstance(final_solution, dict) and final_solution.get('error'):
            return {'solution': '', 'error': final_solution['error']}
        
        return {
            'solution': final_solution,
            'error': None,
            'complexity_regime': ComplexityRegime.HIGH.value,
            'reasoning_approach': 'decomposition_staged',
            'decomposition_results': decomposition_result,
            'staged_solutions': staged_solutions,
            'collapse_prevention': True
        }
    
    async def _decompose_complex_problem(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """複雑問題の分解"""
        # ... (prompt definitions) ...
        response = await self.provider.call(decomposition_prompt, system_prompt, **self.base_model_kwargs)
        return {'decomposition_text': response.get('text', ''), 'error': response.get('error')}
    
    async def _solve_decomposed_problems(
        self, 
        decomposition_result: Dict, 
        original_prompt: str, 
        system_prompt: str
    ) -> List[Dict]:
        """分解された部分問題の段階的解決"""
        # ... (prompt definitions) ...
        response = await self.provider.call(staged_prompt, system_prompt, **self.base_model_kwargs)
        return [{'staged_solution': response.get('text', ''), 'error': response.get('error')}]
    
    async def _integrate_staged_solutions(
        self, 
        staged_solutions: List[Dict], 
        original_prompt: str, 
        system_prompt: str
    ) -> str:
        """段階的解決策の統合"""
        # ... (prompt definitions) ...
        response = await self.provider.call(integration_prompt, system_prompt, **self.base_model_kwargs)
        if response.get('error'):
            return {'solution': '', 'error': response.get('error')}
        return response.get('text', '')