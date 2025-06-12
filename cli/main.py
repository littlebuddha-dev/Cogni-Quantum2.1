# /cli/main.py
# タイトル: CLI main entrypoint with Parallel Mode
# 役割: CLIのエントリーポイントと引数解析。新しい'parallel'モードを追加する。

import argparse
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cli.handler import CogniQuantumCLIV2Fixed
from llm_api.providers import list_providers, list_enhanced_providers
from llm_api.utils.helper_functions import format_json_output, read_from_pipe_or_file

logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(
        description="CogniQuantum V2統合LLM CLI（修正版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("provider", nargs='?', help="使用するLLMプロバイダー")
    parser.add_argument("prompt", nargs='?', default=None, help="LLMへのプロンプト")
    
    # V2専用モードを含む選択肢
    mode_choices = [
        'simple', 'chat', 'reasoning', 'creative-fusion', 'self-correct',
        'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel'
    ]
    parser.add_argument("--mode", default="simple", choices=mode_choices, help="実行モード")
    
    # 基本オプション
    parser.add_argument("--model", help="使用するモデル名")
    parser.add_argument("-f", "--file", help="ファイルからプロンプトを読み込み")
    parser.add_argument("--system-prompt", help="システムプロンプト")
    parser.add_argument("--temperature", type=float, help="生成の多様性")
    parser.add_argument("--max-tokens", type=int, help="最大トークン数")
    parser.add_argument("--json", action="store_true", help="JSON出力")
    
    # 診断・管理オプション
    parser.add_argument("--list-providers", action="store_true", help="プロバイダー一覧表示")
    parser.add_argument("--system-status", action="store_true", help="システム状態表示")
    parser.add_argument("--health-check", action="store_true", help="健全性チェック実行")
    parser.add_argument("--troubleshooting", action="store_true", help="トラブルシューティングガイド")
    
    # V2専用オプション
    v2_group = parser.add_argument_group('V2 Options')
    v2_group.add_argument("--force-v2", action="store_true", help="V2機能強制使用")
    v2_group.add_argument("--no-fallback", action="store_true", help="フォールバック無効")
    v2_group.add_argument("--no-real-time-adjustment", dest="real_time_adjustment", action="store_false", help="リアルタイム複雑性調整を無効化")

    # RAGオプション
    rag_group = parser.add_argument_group('RAG Options')
    rag_group.add_argument("--rag", dest="use_rag", action="store_true", help="RAG機能を有効化")
    rag_group.add_argument("--knowledge-base", dest="knowledge_base_path", help="RAGが使用するナレッジベースのファイルパスまたはURL")
    rag_group.add_argument("--wikipedia", dest="use_wikipedia", action="store_true", help="RAG機能でWikipediaを知識源として使用")

    args = parser.parse_args()

    if args.use_rag and args.use_wikipedia and args.knowledge_base_path:
        parser.error("--knowledge-base と --wikipedia は同時に使用できません。")
    if args.use_rag and not (args.use_wikipedia or args.knowledge_base_path):
        parser.error("--rag を使用するには --knowledge-base または --wikipedia の指定が必要です。")
    if (args.use_wikipedia or args.knowledge_base_path) and not args.use_rag:
         parser.error("--knowledge-base または --wikipedia を使用するには --rag の指定も必要です。")

    cli = CogniQuantumCLIV2Fixed()

    if args.list_providers:
        print("標準プロバイダー:", ", ".join(list_providers()))
        enhanced_info = list_enhanced_providers()
        print("拡張プロバイダー V2:", ", ".join(enhanced_info['v2']))
        return

    if args.system_status:
        cli.print_system_status()
        return

    if args.troubleshooting:
        cli.print_troubleshooting_guide()
        return

    if not args.provider:
        parser.print_help()
        return
        
    is_available = True
    if args.provider == 'ollama':
        ollama_health = await cli._check_ollama_models()
        if not ollama_health.get('server_available') or not ollama_health.get('models_loaded'):
            is_available = False
    else:
        key_map = {'openai': 'OPENAI_API_KEY', 'claude': 'CLAUDE_API_KEY', 'gemini': 'GEMINI_API_KEY', 'huggingface': 'HF_TOKEN'}
        env_var = key_map.get(args.provider)
        if env_var and not os.getenv(env_var):
            is_available = False
    
    if not is_available:
        print("就寝中です・・・")
        return

    if args.health_check:
        try:
            health_report = await cli.check_system_health(args.provider)
            print(format_json_output(health_report) if args.json else json.dumps(health_report, indent=2, ensure_ascii=False))
            return
        except Exception as e:
            print(f"健全性チェック中にエラー: {e}")
            return

    prompt = await read_from_pipe_or_file(args.prompt, args.file)
    if not prompt:
        parser.error("プロンプトが指定されていません。")

    kwargs = {
        'mode': args.mode,
        'system_prompt': args.system_prompt or "",
        'force_v2': args.force_v2,
        'no_fallback': args.no_fallback,
        'use_rag': args.use_rag, 
        'knowledge_base_path': args.knowledge_base_path,
        'use_wikipedia': args.use_wikipedia,
        'real_time_adjustment': args.real_time_adjustment,
    }
    
    if args.model: kwargs['model'] = args.model
    if args.temperature is not None: kwargs['temperature'] = args.temperature
    if args.max_tokens is not None: kwargs['max_tokens'] = args.max_tokens

    try:
        response = await cli.process_request_with_fallback(
            args.provider, prompt, **kwargs
        )
        
        if args.json:
            print(format_json_output(response))
        else:
            text_output = response.get("text", "")
            print(text_output, end='')
            
            if response.get('image_url'):
                print(f"\n\n関連画像: {response['image_url']}")

            if response.get('error') or response.get('fallback_used') or response.get('version') == 'v2':
                print() 

            if response.get('error'):
                print(f"\n⚠️  エラーが発生しました")
                if response.get('all_errors'):
                    print("詳細エラー:")
                    for i, error in enumerate(response['all_errors'], 1):
                        print(f"  {i}. {error}")
                
                if response.get('suggestions'):
                    print("\n💡 改善提案:")
                    for suggestion in response['suggestions']:
                        print(f"  • {suggestion}")
            
            elif response.get('fallback_used'):
                print(f"\n✓ フォールバック実行: {response.get('fallback_type')}")
                if response.get('original_errors'):
                    print("元のエラー:")
                    for error in response['original_errors']:
                        print(f"  • {error}")
            
            elif response.get('version') == 'v2':
                v2_info = response.get('paper_based_improvements', {})
                print(f"\n📊 V2処理情報:")
                print(f"  複雑性体制: {v2_info.get('regime', 'N/A')}")
                print(f"  推論アプローチ: {v2_info.get('reasoning_approach', 'N/A')}")
                if v2_info.get('overthinking_prevention'):
                    print("  ✓ Overthinking防止有効")
                if v2_info.get('collapse_prevention'):
                    print("  ✓ 崩壊防止機構有効")
                if v2_info.get('real_time_adjustment_active'):
                    print("  ✓ リアルタイム複雑性調整有効")
                if v2_info.get('rag_enabled'):
                    rag_source = "Wikipedia" if v2_info.get('rag_source') == 'wikipedia' else 'Knowledge Base'
                    print(f"  ✓ RAGによる知識拡張有効 (ソース: {rag_source})")

    except KeyboardInterrupt:
        print("\n中断されました。")
    except Exception as e:
        logger.critical(f"予期しない致命的エラー: {e}", exc_info=True)
        print(f"\n予期しない致命的なエラーが発生しました: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())