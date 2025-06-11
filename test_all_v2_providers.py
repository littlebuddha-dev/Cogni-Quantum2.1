# /test_all_v2_providers.py
"""
全V2プロバイダーの総合テストスクリプト
作成されたV2プロバイダーの動作確認と性能測定
"""
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# パスの設定
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V2ProviderTester:
    """V2プロバイダーの総合テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.providers_to_test = ['ollama', 'openai', 'claude', 'gemini', 'huggingface']
        self.v2_modes = ['efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized']
        
    async def run_comprehensive_tests(self):
        """総合テストの実行"""
        print("🚀 CogniQuantum V2 プロバイダー総合テスト開始")
        print("=" * 60)
        
        # 1. システム情報の収集
        await self.collect_system_info()
        
        # 2. プロバイダー健全性チェック
        await self.check_all_providers_health()
        
        # 3. V2機能テスト
        await self.test_v2_features()
        
        # 4. パフォーマンステスト
        await self.run_performance_tests()
        
        # 5. 結果の出力
        self.generate_report()

    async def collect_system_info(self):
        """システム情報の収集"""
        print("\n📊 システム情報を収集中...")
        
        try:
            from llm_api.providers import list_providers, list_enhanced_providers, get_provider_info
            
            self.test_results['system_info'] = {
                'timestamp': time.time(),
                'python_version': sys.version,
                'working_directory': str(project_root),
                'standard_providers': list_providers(),
                'enhanced_providers': list_enhanced_providers(),
                'provider_info': get_provider_info()
            }
            
            print(f"✅ システム情報収集完了")
            print(f"   標準プロバイダー: {len(self.test_results['system_info']['standard_providers'])}")
            print(f"   V1拡張プロバイダー: {len(self.test_results['system_info']['enhanced_providers']['v1'])}")
            print(f"   V2拡張プロバイダー: {len(self.test_results['system_info']['enhanced_providers']['v2'])}")
            
        except Exception as e:
            print(f"❌ システム情報収集失敗: {e}")
            self.test_results['system_info'] = {'error': str(e)}

    async def check_all_providers_health(self):
        """全プロバイダーの健全性チェック"""
        print("\n🏥 プロバイダー健全性チェック中...")
        
        try:
            from llm_api.providers import check_all_providers_health
            
            health_results = check_all_providers_health()
            self.test_results['health_check'] = health_results
            
            summary = health_results['summary']
            print(f"✅ 健全性チェック完了")
            print(f"   チェック総数: {summary['total_checked']}")
            print(f"   利用可能: {summary['available']}")
            print(f"   V1拡張: {summary['enhanced_v1']}")
            print(f"   V2拡張: {summary['enhanced_v2']}")
            print(f"   API利用可能: {summary['api_available']}")
            
            # 詳細情報の表示
            for provider_name, provider_health in health_results['providers'].items():
                status_indicators = []
                if provider_health.get('standard', {}).get('available'):
                    status_indicators.append("標準✅")
                if provider_health.get('enhanced_v1', {}).get('available'):
                    status_indicators.append("V1✅")
                if provider_health.get('enhanced_v2', {}).get('available'):
                    status_indicators.append("V2✅")
                    
                status = " ".join(status_indicators) if status_indicators else "❌"
                print(f"   {provider_name}: {status}")
            
        except Exception as e:
            print(f"❌ 健全性チェック失敗: {e}")
            self.test_results['health_check'] = {'error': str(e)}

    async def test_v2_features(self):
        """V2機能の詳細テスト"""
        print("\n🧪 V2機能テスト中...")
        
        self.test_results['v2_features'] = {}
        
        # 利用可能なV2プロバイダーのテスト
        v2_providers = self.test_results.get('system_info', {}).get('enhanced_providers', {}).get('v2', [])
        
        for provider_name in v2_providers:
            print(f"\n🔍 {provider_name} V2機能テスト開始...")
            
            provider_results = {
                'modes_tested': {},
                'features_verified': {},
                'errors': []
            }
            
            # 各V2モードのテスト
            for mode in self.v2_modes:
                try:
                    result = await self.test_provider_mode(provider_name, mode)
                    provider_results['modes_tested'][mode] = result
                    
                    if result['success']:
                        print(f"   ✅ {mode}モード: 成功")
                    else:
                        print(f"   ❌ {mode}モード: {result.get('error', '不明なエラー')}")
                        
                except Exception as e:
                    error_msg = f"{mode}モードテスト中にエラー: {e}"
                    provider_results['errors'].append(error_msg)
                    print(f"   ⚠️  {mode}モード: エラー ({e})")
            
            # V2特有機能の検証
            await self.verify_v2_specific_features(provider_name, provider_results)
            
            self.test_results['v2_features'][provider_name] = provider_results

    async def test_provider_mode(self, provider_name: str, mode: str) -> Dict[str, Any]:
        """特定のプロバイダーとモードをテスト"""
        
        test_prompts = {
            'efficient': "2+2の答えは？",
            'balanced': "気候変動の主要な原因を3つ説明してください",
            'decomposed': "複雑なWebアプリケーションのアーキテクチャを設計してください",
            'adaptive': "この問題を最適な方法で解決してください：効率的な在庫管理システム",
            'paper_optimized': "人工知能の倫理的課題について包括的に論じてください"
        }
        
        prompt = test_prompts.get(mode, "テスト用の基本的な質問です")
        
        try:
            from llm_api.providers import get_provider
            
            # V2プロバイダーの取得
            provider = get_provider(provider_name, enhanced=True, prefer_v2=True)
            
            start_time = time.time()
            
            # モードを指定してテスト
            response = await provider.call(
                prompt,
                mode=mode,
                temperature=0.5,
                max_tokens=500
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': not response.get('error', False),
                'response_length': len(response.get('text', '')),
                'execution_time': execution_time,
                'enhanced': response.get('enhanced', False),
                'version': response.get('version'),
                'v2_improvements': response.get('paper_based_improvements', {}),
                'provider_specific': response.get(f'{provider_name}_specific', {}),
                'mode': mode
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mode': mode
            }

    async def verify_v2_specific_features(self, provider_name: str, provider_results: Dict[str, Any]):
        """V2特有機能の検証"""
        
        features_to_verify = [
            'overthinking_prevention',
            'collapse_prevention', 
            'complexity_regime',
            'reasoning_approach'
        ]
        
        verified_features = {}
        
        for mode, result in provider_results['modes_tested'].items():
            if result.get('success') and result.get('v2_improvements'):
                v2_improvements = result['v2_improvements']
                
                for feature in features_to_verify:
                    if feature in v2_improvements and v2_improvements[feature] is not None:
                        if feature not in verified_features:
                            verified_features[feature] = []
                        verified_features[feature].append(mode)
        
        provider_results['features_verified'] = verified_features
        
        # 検証結果の表示
        for feature, modes in verified_features.items():
            print(f"   📋 {feature}: {', '.join(modes)}で確認")

    async def run_performance_tests(self):
        """パフォーマンステスト"""
        print("\n⚡ パフォーマンステスト中...")
        
        self.test_results['performance'] = {}
        
        # 利用可能なプロバイダーでのパフォーマンス比較
        v2_providers = self.test_results.get('system_info', {}).get('enhanced_providers', {}).get('v2', [])
        
        test_scenarios = [
            {
                'name': 'simple_task',
                'prompt': "Hello, how are you?",
                'mode': 'efficient',
                'expected_tokens': 50
            },
            {
                'name': 'reasoning_task', 
                'prompt': "Explain the concept of machine learning in simple terms",
                'mode': 'balanced',
                'expected_tokens': 200
            },
            {
                'name': 'complex_task',
                'prompt': "Design a sustainable urban transportation system",
                'mode': 'decomposed', 
                'expected_tokens': 500
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📈 シナリオ '{scenario['name']}' テスト中...")
            scenario_results = {}
            
            for provider_name in v2_providers:
                try:
                    # 利用可能性チェック
                    if not await self.is_provider_available(provider_name):
                        print(f"   ⚠️  {provider_name}: スキップ（利用不可）")
                        continue
                    
                    result = await self.test_provider_mode(
                        provider_name, 
                        scenario['mode']
                    )
                    
                    if result['success']:
                        efficiency_score = self.calculate_efficiency_score(
                            result, scenario['expected_tokens']
                        )
                        
                        scenario_results[provider_name] = {
                            'execution_time': result['execution_time'],
                            'response_length': result['response_length'], 
                            'efficiency_score': efficiency_score,
                            'v2_features_used': len(result.get('v2_improvements', {}))
                        }
                        
                        print(f"   ✅ {provider_name}: {result['execution_time']:.2f}s, 効率性: {efficiency_score:.2f}")
                    else:
                        print(f"   ❌ {provider_name}: 失敗")
                        
                except Exception as e:
                    print(f"   ⚠️  {provider_name}: エラー ({e})")
            
            self.test_results['performance'][scenario['name']] = scenario_results

    async def is_provider_available(self, provider_name: str) -> bool:
        """プロバイダーの利用可能性をチェック"""
        health_info = self.test_results.get('health_check', {}).get('providers', {}).get(provider_name, {})
        v2_health = health_info.get('enhanced_v2', {})
        return v2_health.get('available', False)

    def calculate_efficiency_score(self, result: Dict[str, Any], expected_tokens: int) -> float:
        """効率性スコアの計算"""
        execution_time = result.get('execution_time', 1.0)
        response_length = result.get('response_length', 0)
        
        # 基本効率性（文字数/時間）
        basic_efficiency = response_length / execution_time if execution_time > 0 else 0
        
        # 期待値との比較
        length_ratio = min(response_length / expected_tokens, 2.0) if expected_tokens > 0 else 1.0
        
        # V2機能ボーナス
        v2_bonus = 1.2 if result.get('enhanced') and result.get('version') == 'v2' else 1.0
        
        return basic_efficiency * length_ratio * v2_bonus

    def generate_report(self):
        """最終レポートの生成"""
        print("\n" + "=" * 60)
        print("📊 総合テスト結果レポート")
        print("=" * 60)
        
        # システム概要
        system_info = self.test_results.get('system_info', {})
        if 'error' not in system_info:
            print(f"\n🖥️  システム概要:")
            print(f"   V2対応プロバイダー: {len(system_info.get('enhanced_providers', {}).get('v2', []))}")
            print(f"   総プロバイダー数: {len(system_info.get('standard_providers', []))}")
        
        # 健全性サマリー
        health_check = self.test_results.get('health_check', {})
        if 'error' not in health_check:
            summary = health_check.get('summary', {})
            print(f"\n🏥 健全性サマリー:")
            print(f"   V2拡張プロバイダー: {summary.get('enhanced_v2', 0)}/{len(self.providers_to_test)}")
            print(f"   API利用可能: {summary.get('api_available', 0)}")
        
        # V2機能サマリー
        v2_features = self.test_results.get('v2_features', {})
        if v2_features:
            print(f"\n🧪 V2機能テスト結果:")
            for provider, results in v2_features.items():
                successful_modes = sum(1 for result in results['modes_tested'].values() if result.get('success'))
                total_modes = len(results['modes_tested'])
                print(f"   {provider}: {successful_modes}/{total_modes} モード成功")
                
                # V2機能の確認状況
                features = results.get('features_verified', {})
                if features:
                    print(f"     確認済みV2機能: {', '.join(features.keys())}")
        
        # パフォーマンス結果
        performance = self.test_results.get('performance', {})
        if performance:
            print(f"\n⚡ パフォーマンス結果:")
            for scenario, results in performance.items():
                if results:
                    best_provider = max(results.items(), key=lambda x: x[1]['efficiency_score'])
                    print(f"   {scenario}: {best_provider[0]} が最高効率 (スコア: {best_provider[1]['efficiency_score']:.2f})")
        
        # 推奨事項
        self.generate_recommendations()
        
        # JSONレポートの保存
        self.save_json_report()

    def generate_recommendations(self):
        """推奨事項の生成"""
        print(f"\n💡 推奨事項:")
        
        # V2プロバイダーの利用状況に基づく推奨
        v2_providers = self.test_results.get('system_info', {}).get('enhanced_providers', {}).get('v2', [])
        working_providers = []
        
        v2_features = self.test_results.get('v2_features', {})
        for provider, results in v2_features.items():
            successful_modes = sum(1 for result in results['modes_tested'].values() if result.get('success'))
            if successful_modes > 0:
                working_providers.append(provider)
        
        if working_providers:
            print(f"   ✅ 推奨V2プロバイダー: {', '.join(working_providers)}")
        
        # 最適モードの推奨
        mode_success_rates = {}
        for provider, results in v2_features.items():
            for mode, result in results['modes_tested'].items():
                if mode not in mode_success_rates:
                    mode_success_rates[mode] = {'success': 0, 'total': 0}
                mode_success_rates[mode]['total'] += 1
                if result.get('success'):
                    mode_success_rates[mode]['success'] += 1
        
        reliable_modes = [
            mode for mode, stats in mode_success_rates.items()
            if stats['total'] > 0 and stats['success'] / stats['total'] > 0.5
        ]
        
        if reliable_modes:
            print(f"   🎯 信頼性の高いモード: {', '.join(reliable_modes)}")
        
        # セットアップの推奨
        health_check = self.test_results.get('health_check', {})
        if 'error' not in health_check:
            summary = health_check.get('summary', {})
            if summary.get('api_available', 0) < len(v2_providers):
                print(f"   🔧 APIキー設定の確認を推奨")

    def save_json_report(self):
        """JSONレポートの保存"""
        try:
            report_file = project_root / "v2_test_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 詳細レポートを保存: {report_file}")
        except Exception as e:
            print(f"\n❌ レポート保存失敗: {e}")

async def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CogniQuantum V2プロバイダー総合テスト")
    parser.add_argument("--providers", nargs='+', help="テストするプロバイダーを指定")
    parser.add_argument("--modes", nargs='+', help="テストするモードを指定") 
    parser.add_argument("--quick", action="store_true", help="クイックテスト（一部機能のみ）")
    parser.add_argument("--json-only", action="store_true", help="JSON出力のみ")
    
    args = parser.parse_args()
    
    tester = V2ProviderTester()
    
    # カスタマイズ
    if args.providers:
        tester.providers_to_test = args.providers
    if args.modes:
        tester.v2_modes = args.modes
    
    # テスト実行
    if args.quick:
        print("🚀 クイックテストモード")
        await tester.collect_system_info()
        await tester.check_all_providers_health()
    else:
        await tester.run_comprehensive_tests()
    
    # 出力制御
    if args.json_only:
        print(json.dumps(tester.test_results, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    asyncio.run(main())