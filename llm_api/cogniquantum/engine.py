# /llm_api/cogniquantum/engine.py
# タイトル: Enhanced Reasoning Engine with Parallel Sub-problem Solving
# 役割: 論文の発見に基づく改良推論エンジン。高複雑性モードにおいて、サブ問題を並列解決する機能を追加して高速化する。

import logging
import json
import re
import asyncio
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
            # force_regimeが渡された場合でも、regimeがNoneなら分析を行う
            if regime is None:
                complexity_score, regime = self.complexity_analyzer.analyze_complexity(prompt)
        
        logger.info(f"推論実行開始: 体制={regime.value}, 複雑性={complexity_score or 'N/A'}")
        
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
            'error': response.get('error'),
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
            'error': response.get('error'),
            'complexity_regime': ComplexityRegime.MEDIUM.value,
            'reasoning_approach': 'structured_progressive',
            'stage_verification': True
        }
    
    async def _execute_high_complexity_reasoning(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """高複雑性問題の推論（崩壊回避戦略）- サブ問題の並列解決を導入"""
        logger.info("高複雑性推論モード: 分解・並列解決・統合")
        
        sub_problems = await self._decompose_complex_problem(prompt, system_prompt)
        if isinstance(sub_problems, dict) and sub_problems.get('error'):
            return {'solution': '', 'error': sub_problems['error']}
        
        if not sub_problems:
            logger.warning("問題の分解に失敗、またはサブ問題がありません。標準的なアプローチにフォールバックします。")
            return await self._execute_medium_complexity_reasoning(prompt, system_prompt)

        staged_solutions = await self._solve_decomposed_problems(sub_problems, prompt, system_prompt)
        if any(s.get('error') for s in staged_solutions):
             logger.warning("一部のサブ問題の解決中にエラーが発生しました。")
        
        final_solution = await self._integrate_staged_solutions(staged_solutions, prompt, system_prompt)
        if isinstance(final_solution, dict) and final_solution.get('error'):
            return {'solution': '', 'error': final_solution['error']}
        
        return {
            'solution': final_solution,
            'error': None,
            'complexity_regime': ComplexityRegime.HIGH.value,
            'reasoning_approach': 'decomposition_parallel_solve_integration',
            'decomposition_results': sub_problems,
            'staged_solutions': staged_solutions,
            'collapse_prevention': True
        }
    
    async def _decompose_complex_problem(
        self, 
        prompt: str, 
        system_prompt: str
    ) -> List[str] | Dict:
        """複雑な問題を解決可能なサブ問題のJSONリストに分解する"""
        decomposition_prompt = f"""以下の複雑な問題を、解決可能な独立したサブ問題に分解してください。
思考プロセスを段階的に示し、最終的にサブ問題のリストをJSON配列として出力してください。

問題: {prompt}

出力形式は必ず以下のJSONフォーマットに従ってください。
{{
  "sub_problems": [
    "サブ問題1: ...",
    "サブ問題2: ...",
    "サブ問題3: ..."
  ]
}}
"""
        response = await self.provider.call(decomposition_prompt, system_prompt, **self.base_model_kwargs)
        if response.get('error'):
            return {'error': response['error']}
        
        try:
            response_text = response.get('text', '{}').strip()
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                logger.warning(f"分解応答からJSONを抽出できませんでした。箇条書きとしてパースを試みます。: {response_text}")
                return [line.strip() for line in response_text.split('\n') if line.strip() and (line.strip().startswith('*') or line.strip().startswith('-') or re.match(r'^\d+\.', line.strip()))]

            parsed_json = json.loads(match.group(0))
            sub_problems = parsed_json.get("sub_problems", [])
            if not isinstance(sub_problems, list):
                logger.error("分解結果の'sub_problems'がリスト形式ではありません。")
                return []
            logger.info(f"{len(sub_problems)}個のサブ問題に分解しました。")
            return sub_problems
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"問題の分解結果の解析中にエラー: {e}")
            return []
    
    async def _solve_decomposed_problems(
        self, 
        sub_problems: List[str], 
        original_prompt: str, 
        system_prompt: str
    ) -> List[Dict]:
        """分解されたサブ問題を並列で解決する"""
        logger.info(f"{len(sub_problems)}個のサブ問題を並列解決します。")

        async def solve_task(sub_problem: str, index: int) -> Dict:
            staged_prompt = f"""以下の背景情報と元の問題を踏まえ、指定された「サブ問題」を解決してください。
これは大きな問題の一部です。このサブ問題に集中して、詳細かつ具体的な解決策を提示してください。

# 背景
元の問題: {original_prompt}

# 解決すべきサブ問題
{sub_problem}
"""
            logger.debug(f"サブ問題 {index+1} の解決を開始...")
            response = await self.provider.call(staged_prompt, system_prompt, **self.base_model_kwargs)
            logger.debug(f"サブ問題 {index+1} の解決が完了。")
            return {'sub_problem': sub_problem, 'solution': response.get('text', ''), 'error': response.get('error')}

        tasks = [solve_task(sp, i) for i, sp in enumerate(sub_problems)]
        solved_parts = await asyncio.gather(*tasks)
        return solved_parts
    
    async def _integrate_staged_solutions(
        self, 
        staged_solutions: List[Dict], 
        original_prompt: str, 
        system_prompt: str
    ) -> str | Dict:
        """段階的解決策を統合し、一貫性のある最終解を生成する"""
        if not staged_solutions: return ""

        context = "\n\n".join(
            f"### サブ問題: {s['sub_problem']}\n解決策: {s['solution']}"
            for s in staged_solutions if not s.get('error') and s.get('solution')
        )
        integration_prompt = f"""以下の「元の問題」と、それに関連する複数の「サブ問題の解決策」を統合し、一貫性のある包括的な最終解答を作成してください。

# 元の問題
{original_prompt}

# サブ問題の解決策
---
{context}
---

# 最終解答
上記の情報を synthesis (統合・総合) し、論理的な流れを持つ、最終的な答えを生成してください。
"""
        response = await self.provider.call(integration_prompt, system_prompt, **self.base_model_kwargs)
        if response.get('error'):
            return {'error': response['error']}
        return response.get('text', '')