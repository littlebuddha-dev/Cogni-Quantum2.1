# /llm_api/tools/image_retrieval.py
# パス: littlebuddha-dev/cogni-quantum2.1/Cogni-Quantum2.1-fb17e3467b051803511a1506de5e02910bbae07e/llm_api/tools/image_retrieval.py
# タイトル: Image Retrieval Tool
# 役割: SerpApiを利用してWebから画像を検索し、結果を返すツール。

import logging
import os
from typing import NamedTuple, Optional
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)

class ImageResult(NamedTuple):
    """Represents a single image search result."""
    title: str
    source: str
    content_url: str
    thumbnail_url: str

def search(query: str) -> Optional[ImageResult]:
    """Performs an image search using SerpApi and returns the top result."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.warning("SERPAPI_API_KEY環境変数が設定されていません。画像検索はスキップされます。")
        return None

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "tbm": "isch", # Specify image search
    }

    try:
        search_client = GoogleSearch(params)
        results = search_client.get_dict()
        
        image_results = results.get("images_results")
        if not image_results:
            logger.warning(f"画像検索で結果が見つかりませんでした: '{query}'")
            return None
        
        # Return the first result
        top_result = image_results[0]
        return ImageResult(
            title=top_result.get("title"),
            source=top_result.get("source"),
            content_url=top_result.get("original"),
            thumbnail_url=top_result.get("thumbnail"),
        )
    except Exception as e:
        logger.error(f"SerpApi経由の画像検索に失敗しました: {e}", exc_info=True)
        return None