# /llm_api/cogniquantum/system.py
"""
CogniQuantumシステムV2本体
"""
import logging
from typing import Any, Dict, Optional

from .engine import EnhancedReasoningEngine
from .enums import ComplexityRegime
from .tracker import SolutionTracker, ReasoningMetrics
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

class CogniQuantumSystemV2:
    """
    論文「The Illusion of Thinking」の発見に基づく改良版CogniQuantumシステム
    """
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any]):
        logger.info("CogniQuantumシステムV2を初期化中（論文知見ベース改善版）")
        if not provider:
            raise ValueError("有効なLLMプロバイダーがCogniQuantumSystemV2に必要です。")
            
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        self.reasoning_engine = EnhancedReasoningEngine(provider, base_model_kwargs)
        self.solution_tracker = SolutionTracker()
        self.max_refinement_cycles = 2
        self.token_budget_management = True
        self.overthinking_prevention = True
        logger.info("CogniQuantumシステムV2の初期化完了")
    
    async def solve_problem(
        self,
        prompt: str,
        system_prompt: str = "",
        force_regime: Optional[ComplexityRegime] = None
    ) -> Dict[str, Any]:
        """論文知見に基づく問題解決プロセス"""
        logger.info(f"問題解決プロセス開始（V2）: {prompt[:80]}...")
        try:
            complexity_score, regime = self.reasoning_engine.complexity_analyzer.analyze_complexity(prompt)
            if force_regime:
                regime = force_regime
                logger.info(f"レジームを '{regime.value}' に強制設定しました。")

            reasoning_result = await self.reasoning_engine.execute_reasoning(
                prompt, system_prompt, complexity_score, regime
            )
            
            # 推論エンジンからのエラーをチェック
            if reasoning_result.get('error'):
                return {'success': False, 'error': reasoning_result['error']}

            final_result = await self._evaluate_and_refine(
                reasoning_result, prompt, system_prompt, regime
            )
            metrics = self._collect_metrics(complexity_score, regime, reasoning_result)
            
            return {
                'success': True,
                'final_solution': final_result['solution'],
                'complexity_analysis': {
                    'complexity_score': complexity_score,
                    'regime': regime.value,
                    'reasoning_approach': reasoning_result.get('reasoning_approach')
                },
                'performance_metrics': metrics,
                'v2_improvements': {
                    'overthinking_prevention': reasoning_result.get('overthinking_prevention', False),
                    'collapse_prevention': reasoning_result.get('collapse_prevention', False),
                    'algorithm_assistance': reasoning_result.get('algorithm_assistance', False)
                }
            }
        except Exception as e:
            logger.error(f"問題解決中にエラーが発生しました（V2）: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'version': 'v2'}
    
    async def _evaluate_and_refine(
        self,
        reasoning_result: Dict[str, Any],
        original_prompt: str,
        system_prompt: str,
        regime: ComplexityRegime
    ) -> Dict[str, Any]:
        """結果の評価と必要に応じた洗練"""
        if regime == ComplexityRegime.LOW:
            logger.info("低複雑性問題: refinementスキップ（overthinking防止）")
            return reasoning_result
        
        if regime in [ComplexityRegime.MEDIUM, ComplexityRegime.HIGH]:
            refined_solution = await self._perform_limited_refinement(
                reasoning_result['solution'], original_prompt, system_prompt
            )
            reasoning_result['solution'] = refined_solution
        
        return reasoning_result
    
    async def _perform_limited_refinement(
        self,
        solution: str,
        original_prompt: str,
        system_prompt: str
    ) -> str:
        """限定的洗練（論文発見に基づく制約付き）"""
        refinement_prompt = f"""以下の解答を簡潔に検証し、必要最小限の改善のみを行ってください。
過度な変更や追加分析は避けてください。

元の問題: {original_prompt}

現在の解答: {solution}

検証ポイント:
1. 論理的一貫性の確認
2. 明らかな誤りの修正
3. 不足している重要要素の補完

重要: 必要最小限の変更に留め、解答の核心的価値を保持してください。"""
        
        response = await self.provider.call(refinement_prompt, system_prompt, **self.base_model_kwargs)
        if isinstance(response, dict) and response.get('error'):
            return solution # 洗練でエラーが起きても元の解を返す
        return response.get('text', solution)
    
    def _collect_metrics(
        self,
        complexity_score: float,
        regime: ComplexityRegime,
        reasoning_result: Dict[str, Any]
    ) -> ReasoningMetrics:
        """推論プロセスのメトリクス収集"""
        return ReasoningMetrics(
            complexity_score=complexity_score,
            regime=regime,
            thinking_tokens_used=len(reasoning_result.get('solution', '').split()),
            solution_positions=[],
            correct_solution_positions=[],
            first_correct_position=None,
            overthinking_detected=reasoning_result.get('overthinking_prevention', False),
            consistency_score=0.85  # 暫定値
        )