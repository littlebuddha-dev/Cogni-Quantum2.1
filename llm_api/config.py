# /llm_api/config.py
# タイトル: Centralized Settings Management with Concurrency Limit
# 役割: プロジェクト全体の設定を管理する。Ollamaの同時実行数制限を追加。

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    プロジェクト全体の設定を管理するクラス。
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    # SERPAPI_API_KEY は削除済み

    # --- Provider Defaults ---
    OLLAMA_API_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: float = 600.0
    # ★★★ 追加箇所 ★★★
    # Ollamaへの同時リクエスト数の上限。PCのスペックに応じて調整可能。
    OLLAMA_CONCURRENCY_LIMIT: int = 2

    # --- Default Models ---
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
    CLAUDE_DEFAULT_MODEL: str = "claude-3-haiku-20240307"
    GEMINI_DEFAULT_MODEL: str = "gemini-1.5-flash-latest"
    HUGGINGFACE_DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    OLLAMA_DEFAULT_MODEL: str = "gemma3:latest"

    # --- CogniQuantum V2 Settings ---
    V2_DEFAULT_MODE: str = "adaptive"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"


settings = Settings()