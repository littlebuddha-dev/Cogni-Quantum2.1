# /llm_api/cogniquantum/system.py
# タイトル: CogniQuantum System with Parallel Pipelines & Real-time Adjustment
# 役割: CogniQuantumシステム本体。新しく'parallel'モードを追加し、複数の思考パイプラインを並列実行して最良の結果を選択する機能を追加する。

import logging
import json
import re
import asyncio
from typing import Any, Dict, Optional, List

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
        self.max_refinement_cycles = 1
        self.max_adjustment_attempts = 2
        logger.info("CogniQuantumシステムV2の初期化完了")
    
    async def _search_for_image(self, original_prompt: str, final_solution: str) -> Optional[str]:
        """元のプロンプトと最終解答に基づいて関連画像を検索する"""
        image_keywords = ['画像', '写真', '絵', 'イラスト', '表示して']
        if not any(keyword in original_prompt for keyword in image_keywords):
            return None
        logger.info("画像検索のキーワードが検出されました。関連画像を検索します。")
        query_generation_prompt = f"""以下の会話と最終回答に基づいて、この文脈に最も合う画像を検索するための、簡潔で具体的な英語の検索クエリを生成してください。\n\n元の質問: {original_prompt}\n最終回答: {final_solution}\n\n検索クエリ(英語):"""
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

    async def _self_evaluate_solution(self, solution: str, original_prompt: str, current_regime: ComplexityRegime) -> Dict[str, Any]:
        """LLM自身に現在の解を評価させ、次のステップを判断させる"""
        if current_regime == ComplexityRegime.HIGH:
            return {"is_sufficient": True}
        evaluation_prompt = f"""以下の「質問」とそれに対する「現在の回答」を評価してください。\n回答が質問の意図を十分に満たしているか、あるいは表面的・不十分であるかを判断してください。\n\n判断基準:\n- 質問の全ての要素に答えているか？\n- 回答は具体的か、それとも一般的すぎるか？\n- より深い分析や詳細な説明が必要か？\n\n質問: {original_prompt}\n現在の回答: {solution}\n\n評価結果を以下のJSON形式で一行で出力してください。出力はJSONオブジェクトのみとし、前後に他のテキストを含めないでください。\n{{\n  "is_sufficient": boolean,\n  "reason": "回答が十分かどうかの簡単な理由",\n  "next_recommended_complexity": "{'medium' if current_regime == ComplexityRegime.LOW else 'high'}"\n}}"""
        try:
            eval_kwargs = self.base_model_kwargs.copy()
            eval_kwargs['temperature'] = 0.1
            response = await self.provider.call(evaluation_prompt, "You are a solution evaluator.", **eval_kwargs)
            response_text = response.get('text', '{}').strip()
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                logger.warning(f"自己評価の応答からJSONを抽出できませんでした。応答: {response_text}")
                return {"is_sufficient": True}
            parsed_json = json.loads(match.group(0))
            is_sufficient = parsed_json.get("is_sufficient", True)
            if not isinstance(is_sufficient, bool): is_sufficient = True
            if is_sufficient:
                logger.info("自己評価: 現在の解は十分です。")
                return {"is_sufficient": True}
            else:
                logger.info(f"自己評価: 現在の解は不十分です。理由: {parsed_json.get('reason')}")
                next_regime_str = parsed_json.get("next_recommended_complexity", current_regime.value)
                next_regime = ComplexityRegime(next_regime_str)
                return {"is_sufficient": False, "next_regime": next_regime}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"自己評価の解析中にエラー: {e}", exc_info=True)
            return {"is_sufficient": True}

    async def solve_problem(
        self,
        prompt: str,
        system_prompt: str = "",
        force_regime: Optional[ComplexityRegime] = None,
        use_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        use_wikipedia: bool = False,
        real_time_adjustment: bool = True,
        mode: str = 'adaptive'
    ) -> Dict[str, Any]:
        """論文知見に基づく問題解決プロセス"""
        # ログメッセージを修正
        logger.info(f"問題解決プロセス開始（V2, モード: {mode}）: {prompt[:80]}...")
        
        if mode == 'parallel':
            return await self._execute_parallel_pipelines(prompt, system_prompt, use_rag, knowledge_base_path, use_wikipedia)

        current_prompt = prompt
        rag_source = None
        if use_rag:
            logger.info("RAGが有効です。")
            rag_manager = RAGManager(provider=self.provider, use_wikipedia=use_wikipedia, knowledge_base_path=knowledge_base_path)
            current_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'
        try:
            complexity_score, current_regime = self.reasoning_engine.complexity_analyzer.analyze_complexity(current_prompt)
            if force_regime:
                current_regime = force_regime
                logger.info(f"レジームを '{current_regime.value}' に強制設定しました。")
            reasoning_result = None
            for attempt in range(self.max_adjustment_attempts):
                logger.info(f"推論試行 {attempt + 1}/{self.max_adjustment_attempts} (レジーム: {current_regime.value})")
                reasoning_result = await self.reasoning_engine.execute_reasoning(current_prompt, system_prompt, complexity_score, current_regime)
                if reasoning_result.get('error'):
                    return {'success': False, 'error': reasoning_result['error']}
                final_solution = reasoning_result.get('solution')
                if force_regime or not real_time_adjustment or (attempt + 1) >= self.max_adjustment_attempts:
                    break
                evaluation = await self._self_evaluate_solution(final_solution, prompt, current_regime)
                if evaluation.get("is_sufficient"):
                    break
                else:
                    new_regime = evaluation.get("next_regime", current_regime)
                    if new_regime.value != current_regime.value:
                        logger.info(f"複雑性を再調整: {current_regime.value} -> {new_regime.value}")
                        current_regime = new_regime
                        current_prompt = f"前回の回答は不十分でした。より深く、包括的な分析を行ってください。\n元の質問: {prompt}\n前回の回答: {final_solution}\n"
                    else:
                        logger.info("同じ複雑性レジームが推奨されたため、調整を終了します。")
                        break
            final_result = await self._evaluate_and_refine(reasoning_result, current_prompt, system_prompt, current_regime)
            metrics = self._collect_metrics(complexity_score, current_regime, reasoning_result)
            image_url = await self._search_for_image(prompt, final_result['solution'])
            v2_improvements = {
                'overthinking_prevention': reasoning_result.get('overthinking_prevention', False),
                'collapse_prevention': reasoning_result.get('collapse_prevention', False),
                'algorithm_assistance': reasoning_result.get('algorithm_assistance', False),
                'rag_enabled': use_rag,
                'rag_source': rag_source,
                'real_time_adjustment_active': real_time_adjustment and not force_regime,
            }
            return {
                'success': True,
                'final_solution': final_result['solution'],
                'image_url': image_url,
                'complexity_analysis': {'complexity_score': complexity_score, 'regime': current_regime.value, 'reasoning_approach': reasoning_result.get('reasoning_approach')},
                'performance_metrics': metrics,
                'v2_improvements': v2_improvements,
                'version': 'v2'
            }
        except Exception as e:
            logger.error(f"問題解決中にエラーが発生しました（V2）: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'version': 'v2'}

    async def _execute_parallel_pipelines(self, prompt: str, system_prompt: str, use_rag: bool, knowledge_base_path: Optional[str], use_wikipedia: bool) -> Dict[str, Any]:
        """3つの主要な思考パイプラインを並列実行し、最良の結果を選択する"""
        logger.info("並列推論パイプライン実行開始: efficient, balanced, decomposed")
        final_prompt = prompt
        rag_source = None
        if use_rag:
            rag_manager = RAGManager(provider=self.provider, use_wikipedia=use_wikipedia, knowledge_base_path=knowledge_base_path)
            final_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'
        tasks = [
            self.reasoning_engine.execute_reasoning(final_prompt, system_prompt, regime=ComplexityRegime.LOW),
            self.reasoning_engine.execute_reasoning(final_prompt, system_prompt, regime=ComplexityRegime.MEDIUM),
            self.reasoning_engine.execute_reasoning(final_prompt, system_prompt, regime=ComplexityRegime.HIGH),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_solutions = [res for res in results if not isinstance(res, Exception) and not res.get('error')]
        if not valid_solutions:
            return {'success': False, 'error': "全ての並列パイプラインが失敗しました。", 'version': 'v2'}
        best_solution_info = await self._select_best_solution(valid_solutions, prompt)
        final_solution = best_solution_info['solution']
        image_url = await self._search_for_image(prompt, final_solution)
        v2_improvements = {
            'reasoning_approach': f"parallel_best_of_{len(valid_solutions)}",
            'selected_regime': best_solution_info.get('complexity_regime'),
            'rag_enabled': use_rag,
            'rag_source': rag_source,
        }
        return {
            'success': True,
            'final_solution': final_solution,
            'image_url': image_url,
            'complexity_analysis': {'regime': 'parallel', 'reasoning_approach': v2_improvements['reasoning_approach']},
            'performance_metrics': None,
            'v2_improvements': v2_improvements,
            'version': 'v2'
        }

    async def _select_best_solution(self, solutions: List[Dict[str, Any]], original_prompt: str) -> Dict[str, Any]:
        """LLMを使って複数の解決策から最良のものを選択する"""
        if len(solutions) == 1:
            return solutions[0]
        solution_texts = [s.get('solution', '') for s in solutions]
        selection_prompt = f"以下の「元の質問」に対して、複数の「回答案」が生成されました。\nそれぞれの回答案を慎重に評価し、最も高品質で、質問の意図を完全に満たすものを1つ選択してください。\n\n# 元の質問\n{original_prompt}\n\n# 回答案\n---\n"
        for i, sol_text in enumerate(solution_texts):
            selection_prompt += f"## 回答案 {i+1} ({solutions[i].get('complexity_regime')})\n{sol_text}\n\n---\n"
        selection_prompt += "\n# あなたのタスク\n全ての回答案を比較検討し、最も優れている回答案の番号をJSON形式で出力してください。\n\n出力形式:\n{{\n  \"best_choice_index\": <選択した回答案のインデックス (0始まり)>,\n  \"reason\": \"その回答案が最も優れていると判断した簡潔な理由\"\n}}\n"
        try:
            eval_kwargs = self.base_model_kwargs.copy()
            eval_kwargs['temperature'] = 0.0
            response = await self.provider.call(selection_prompt, "You are an expert solution evaluator.", **eval_kwargs)
            response_text = response.get('text', '{}').strip()
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                logger.warning("最良解の選択応答からJSONを抽出できませんでした。最初の解を選択します。")
                return solutions[0]
            parsed_json = json.loads(match.group(0))
            best_index = parsed_json.get("best_choice_index", 0)
            if not isinstance(best_index, int) or not (0 <= best_index < len(solutions)):
                logger.warning(f"無効なインデックス {best_index} が返されました。最初の解を選択します。")
                return solutions[0]
            logger.info(f"最良解としてインデックス {best_index} ({solutions[best_index].get('complexity_regime')}) が選択されました。理由: {parsed_json.get('reason')}")
            return solutions[best_index]
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"最良解の選択中にエラー: {e}。最初の解を選択します。")
            return solutions[0]

    async def _evaluate_and_refine(self, reasoning_result: Dict[str, Any], original_prompt: str, system_prompt: str, regime: ComplexityRegime) -> Dict[str, Any]:
        """結果の評価と必要に応じた洗練"""
        if regime == ComplexityRegime.LOW:
            logger.info("低複雑性問題: refinementスキップ（overthinking防止）")
            return reasoning_result
        if regime in [ComplexityRegime.MEDIUM, ComplexityRegime.HIGH]:
            refined_solution = await self._perform_limited_refinement(reasoning_result['solution'], original_prompt, system_prompt)
            reasoning_result['solution'] = refined_solution
        return reasoning_result
    
    async def _perform_limited_refinement(self, solution: str, original_prompt: str, system_prompt: str) -> str:
        """限定的洗練（論文発見に基づく制約付き）"""
        refinement_prompt = f"以下の解答を簡潔に検証し、必要最小限の改善のみを行ってください。\n過度な変更や追加分析は避けてください。\n\n元の問題: {original_prompt}\n\n現在の解答: {solution}\n\n検証ポイント:\n1. 論理的一貫性の確認\n2. 明らかな誤りの修正\n3. 不足している重要要素の補完\n\n重要: 必要最小限の変更に留め、解答の核心的価値を保持してください。"
        response = await self.provider.call(refinement_prompt, system_prompt, **self.base_model_kwargs)
        if isinstance(response, dict) and response.get('error'):
            return solution
        return response.get('text', solution)
    
    def _collect_metrics(self, complexity_score: float, regime: ComplexityRegime, reasoning_result: Dict[str, Any]) -> ReasoningMetrics:
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