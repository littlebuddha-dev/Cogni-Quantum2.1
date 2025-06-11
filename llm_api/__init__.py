# /llm_api/__init__.py
"""
CogniQuantum統合LLM APIモジュール

革新的認知量子推論システムを統合したLLMインターフェース
"""

__version__ = "2.0.0"
__author__ = "CogniQuantum Project"
__description__ = "CogniQuantum統合LLM CLI - 革新的認知量子推論システム"

import logging
import os
from typing import Dict, Any, Optional

# ロギング設定
def setup_logging(level: str = "INFO"):
    """ロギングの設定"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # CogniQuantum特有のロガー
    cq_logger = logging.getLogger('cogniquantum')
    cq_logger.setLevel(log_level)

# 環境変数から設定を読み込み
def load_config() -> Dict[str, Any]:
    """環境変数から設定を読み込む"""
    config = {
        # API キー
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'claude_api_key': os.getenv('CLAUDE_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'hf_token': os.getenv('HF_TOKEN'),
        
        # CogniQuantum設定
        'cogniquantum_enabled': os.getenv('COGNIQUANTUM_ENABLED', 'true').lower() == 'true',
        'complexity_threshold': float(os.getenv('COGNIQUANTUM_COMPLEXITY_THRESHOLD', '1.0')),
        'default_mode': os.getenv('COGNIQUANTUM_DEFAULT_MODE', 'adaptive_thinking'),
        'learning_enabled': os.getenv('COGNIQUANTUM_LEARNING_ENABLED', 'true').lower() == 'true',
        'monitoring_enabled': os.getenv('COGNIQUANTUM_MONITORING_ENABLED', 'true').lower() == 'true',
        
        # プロバイダー設定
        'openai_model': os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4-turbo-preview'),
        'claude_model': os.getenv('CLAUDE_DEFAULT_MODEL', 'claude-3-sonnet-20240229'),
        'gemini_model': os.getenv('GEMINI_DEFAULT_MODEL', 'gemini-pro'),
        'ollama_model': os.getenv('OLLAMA_DEFAULT_MODEL', 'gemma2')
    }
    return config

# モジュール初期化時にロギングを設定
setup_logging()