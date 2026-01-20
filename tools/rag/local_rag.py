# RAG System Tool moved to tools/rag
# Full implementation copied from tools/local_rag.py

import logging
import sys
from typing import Dict, Any

from tools.rag.rag_utils import (
    rewrite_to_single_or_empty,
    summarize_or_empty,
    rag_retrieve
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Supported repositories for RAG system
SUPPORTED_REPOS = ["sympy", "scipy", "numpy", "math", "cmath", "fractions", "itertools"]


def local_rag_retrieve(
    repo_name: str,
    query: str,
    top_k: int = 3,
    rag_base_url: str = None,
    chat_base_url: str = None,
    chat_model: str = None,
    chat_timeout: int = None,
    log_requests: bool = True
) -> Dict[str, Any]:
    if log_requests:
        logger.info(f"local_rag_retrieve called: repo={repo_name}, query='{query[:50]}...'")
    
    try:
        if repo_name not in SUPPORTED_REPOS:
            error_msg = f"Unsupported repository name: {repo_name}. Supported: {SUPPORTED_REPOS}"
            if log_requests:
                logger.warning(error_msg)
            return {
                "text_result": f"Error: {error_msg}",
                "status": "error",
                "error": error_msg
            }
        
        repo_url = f"local/{repo_name}"
        
        rewritten = rewrite_to_single_or_empty(
            query,
            chat_base_url=chat_base_url,
            chat_model=chat_model,
            chat_timeout=chat_timeout
        )
        effective_query = rewritten if isinstance(rewritten, str) and rewritten.strip() else query
        
        if log_requests and rewritten:
            logger.debug(f"Query rewritten: '{query[:30]}...' -> '{rewritten[:30]}...'")
        
        retrieve_result = rag_retrieve(
            repo_url=repo_url,
            query=effective_query,
            top_k=top_k,
            rag_base_url=rag_base_url
        )
        
        if "error" in retrieve_result and retrieve_result.get("error"):
            error_msg = retrieve_result["error"]
            if log_requests:
                logger.warning(f"RAG retrieve failed: {error_msg}")
            return {
                "text_result": f"Error: {error_msg}",
                "status": "error",
                "error": error_msg
            }
        
        context_text = retrieve_result.get("context_text", "")
        
        summary = summarize_or_empty(
            effective_query,
            context_text,
            repo_name,
            chat_base_url=chat_base_url,
            chat_model=chat_model,
            chat_timeout=chat_timeout
        )
        
        if isinstance(summary, str) and summary.strip():
            final_text = summary.strip()
        else:
            final_text = context_text.strip()
        
        if log_requests:
            logger.info(f"local_rag_retrieve: returned {len(final_text)} chars")
            logger.debug(f"local_rag_retrieve result: {final_text[:200]}...")
        
        return {
            "text_result": final_text,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Request failed: {str(e)}"
        if log_requests:
            logger.error(f"local_rag_retrieve error: {error_msg}")
        return {
            "text_result": f"Error: {error_msg}",
            "status": "error",
            "error": error_msg
        }


# Keep test harness similar to original for manual testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test_local_rag():
        print("=" * 60)
        print("Testing RAG System Retrieve")
        print("=" * 60)
        
        print("\n[Test 1] Invalid repository name")
        result1 = local_rag_retrieve(repo_name="invalid_repo", query="test query", log_requests=False)
        success1 = result1["status"] == "error" and "Unsupported" in result1.get("error", "")
        print(f"Status: {result1['status']}")
        print(f"Error: {result1.get('error', 'N/A')}")
        print(f"✓ PASS" if success1 else "✗ FAIL")
        
        print("\n[Test 2] Valid query (requires running services)")
        print("Note: This test may fail if RAG services are not running.")
        result2 = local_rag_retrieve(repo_name="sympy", query="How to solve equations with sympy?", top_k=2, log_requests=True)
        print(f"Status: {result2['status']}")
        if result2["status"] == "success":
            print(f"Result length: {len(result2['text_result'])} chars")
            print(f"Result preview: {result2['text_result'][:200]}...")
        else:
            print(f"Error: {result2.get('error', 'Unknown error')}")
        
        return success1
    
    success = test_local_rag()
    sys.exit(0 if success else 1)
