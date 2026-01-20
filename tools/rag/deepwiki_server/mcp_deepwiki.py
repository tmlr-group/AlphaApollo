"""
DeepWiki MCP Server Implementation - Chat with Repository Only
This module provides a simplified MCP (Model Context Protocol) server implementation for DeepWiki.
It exposes only the chat_with_repository function as an MCP tool, enabling integration with
AI models and applications that support the MCP protocol.

Key Features:
- Repository chat with RAG-powered AI for Python packages (sympy, scipy, numpy, math, cmath, fractions, itertools)

Available Tools:
- chat_with_repository: Chat with repositories using DeepWiki's AI

Dependencies:
- fastmcp: FastMCP framework for MCP server implementation  
- requests: HTTP client for API calls

Usage:
    python tools/deepwiki_server/mcp_deepwiki.py
"""

from fastmcp import FastMCP
import sys
import os
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

mcp = FastMCP("DeepWikiServer")

# DeepWiki API base configuration
DEEPWIKI_BASE_URL = os.environ.get("DEEPWIKI_BASE_URL", "http://localhost:11048")

@mcp.tool()
def chat_with_repository(
    repo_name: str,
    query: str,
    timeout: int = 120
):
    """A search tool for querying Python package documentation (sympy, scipy, numpy, math, cmath, fractions, itertools). Use this tool whenever you are unsure about how to implement specific functionality, syntax, or methods in these packages before attempting to generate Python code. This helps ensure accurate and efficient code by providing relevant documentation snippets, examples, and descriptions to guide your reasoning and implementation.
Args:
    repo_name: Name of the repository to query, e.g. 'sympy'.
    query: query text, e.g., 'Function interface and examples of calling sympy to solve nonlinear equations'
Returns:
    A dictionary with 'status' ('success' or 'error'), 'response' (the AI response),and 'status\_code' (HTTP status code)
"""

    try:
        # Build request data
        if repo_name in ["math", "cmath", "fractions", "itertools"]:
            repo_url = f"local/cpython"
        elif repo_name in ["sympy", "scipy", "numpy"]:
            repo_url = f"local/{repo_name}"
        else:
            return {
                "status": "error",
                "error": f"Unsupported repository name: {repo_name}",
                "status_code": 400
            }


        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        request_data = {
            "repo_url": repo_url,
            "messages": messages,
            'provider': 'openai',
            'model': 'Qwen/Qwen3-4B',
            'language': 'en',
            'type': 'github',
        }

        # Send request to DeepWiki API
        response = requests.post(
            f"{DEEPWIKI_BASE_URL}/chat/completions/stream",
            json=request_data,
            timeout=timeout
        )

        if response.status_code == 200:
            return {
                "status": "success",
                "response": response.text,
                "status_code": response.status_code
            }
        else:
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text}",
                "status_code": response.status_code
            }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Request failed: {str(e)}",
            "status_code": 500
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="sse", choices=["stdio", "streamable-http", "sse", "http"])
    parser.add_argument("--port", default=11045, type=int)
    parser.add_argument("--path", default="/sse", type=str)
    args = parser.parse_args()

    print("\nStart MCP DeepWiki Tool service:")
    print("!"*100)

    if args.transport == "streamable-http":
        mcp.run(transport='streamable-http', port=args.port, path=args.path)
    elif args.transport == "sse":
        mcp.run(transport='sse', port=args.port, path=args.path)
    elif args.transport == "http":
        mcp.run(transport='http', port=args.port, path=args.path)
    else:
        mcp.run(transport='stdio') 