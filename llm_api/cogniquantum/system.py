# /llm_api/cogniquantum/system.py
# パス: littlebuddha-dev/cogni-quantum2.1/Cogni-Quantum2.1-fb17e3467b051803511a1506de5e02910bbae07e/llm_api/cogniquantum/system.py
# タイトル: CogniQuantum System (Final RAG/Image Integration)
# 役割: CogniQuantumシステム本体。RAGManagerにproviderを正しく渡し、RAGと画像検索機能を統合する。

import logging
from typing import Any, Dict, Optional
import re 

from .engine import EnhancedReasoningEngine
from .enums import ComplexityRegime
from .tracker import SolutionTracker, ReasoningMetrics
from ..providers.base import LLMProvider
from ..rag import RAGManager
from ..tools import image_retrieval

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
    
    async def _search_for_image(self, original_prompt: str, final_solution: str) -> Optional[str]:
        """元のプロンプトと最終解答に基づいて関連画像を検索する"""
        image_keywords = ['画像', '写真', '絵', 'イラスト', '表示して']
        if not any(keyword in original_prompt for keyword in image_keywords):
            return None

        logger.info("画像検索のキーワードが検出されました。関連画像を検索します。")
        
        query_generation_prompt = f"""以下の会話と最終回答に基づいて、この文脈に最も合う画像を検索するための、簡潔で具体的な英語の検索クエリを生成してください。

元の質問: {original_prompt}
最終回答: {final_solution}

検索クエリ(英語):"""
        
        try:
            response = await self.provider.call(query_generation_prompt, "", **self.base_model_kwargs)
            image_search_query = response.get('text', '').strip().replace("「", "").replace("」", "").replace("\"", "")

            if not image_search_query:
                logger.warning("画像検索クエリの生成に失敗しました。")
                return None

            logger.info(f"生成された画像検索クエリ: '{image_search_query}'")
            
            image_result = image_retrieval.search(query=image_search_query)
            
            if image_result and image_result.content_url:
                logger.info(f"画像が見つかりました: {image_result.content_url}")
                return image_result.content_url
            else:
                logger.warning("画像検索で画像が見つかりませんでした。")
                return None

        except Exception as e:
            logger.error(f"画像検索プロセス中にエラー: {e}", exc_info=True)
            return None

    async def solve_problem(
        self,
        prompt: str,
        system_prompt: str = "",
        force_regime: Optional[ComplexityRegime] = None,
        use_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        use_wikipedia: bool = False
    ) -> Dict[str, Any]:
        """論文知見に基づく問題解決プロセス"""
        logger.info(f"問題解決プロセス開始（V2）: {prompt[:80]}...")
        
        final_prompt = prompt
        rag_source = None
        
        if use_rag:
            logger.info(f"RAGが有効です。")
            rag_manager = RAGManager(
                provider=self.provider, # RAGManagerにproviderを渡す【重要修正点】
                use_wikipedia=use_wikipedia,
                knowledge_base_path=knowledge_base_path
            )
            final_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'

        try:
            complexity_score, regime = self.reasoning_engine.complexity_analyzer.analyze_complexity(final_prompt)
            if force_regime:
                regime = force_regime
                logger.info(f"レジームを '{regime.value}' に強制設定しました。")

            reasoning_result = await self.reasoning_engine.execute_reasoning(
                final_prompt, system_prompt, complexity_score, regime
            )
            
            if reasoning_result.get('error'):
                return {'success': False, 'error': reasoning_result['error']}

            final_result = await self._evaluate_and_refine(
                reasoning_result, final_prompt, system_prompt, regime
            )
            metrics = self._collect_metrics(complexity_score, regime, reasoning_result)
            
            image_url = await self._search_for_image(prompt, final_result['solution'])

            v2_improvements = {
                'overthinking_prevention': reasoning_result.get('overthinking_prevention', False),
                'collapse_prevention': reasoning_result.get('collapse_prevention', False),
                'algorithm_assistance': reasoning_result.get('algorithm_assistance', False),
                'rag_enabled': use_rag,
                'rag_source': rag_source
            }

            return {
                'success': True,
                'final_solution': final_result['solution'],
                'image_url': image_url,
                'complexity_analysis': {
                    'complexity_score': complexity_score,
                    'regime': regime.value,
                    'reasoning_approach': reasoning_result.get('reasoning_approach')
                },
                'performance_metrics': metrics,
                'v2_improvements': v2_improvements
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
            return solution
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
            consistency_score=0.85 
        )