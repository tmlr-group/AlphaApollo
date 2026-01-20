import json
import logging
import threading
import requests
from urllib.parse import urlparse
from typing import Dict, Any
from tools.core import tool, ToolGroup
from tools.informalmath_verify import call_informalmath_verify
from tools.python_code import execute_python_code
from tools.rag.local_rag import local_rag_retrieve
from tools.agent import Agent

DEFAULT_TIMEOUT = 30

logger = logging.getLogger(__name__)
# TODO: Here we hardcode DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

class InformalMathToolGroup(ToolGroup):
    """
    Tool group for Informal Math environments.
    
    Args:
        log_requests: Whether to log tool requests
        vllm_cfg: VLLM configuration (deprecated, use verifier_cfg)
        verifier_cfg: Verifier agent configuration
        tool_config: Dict with tool configuration. Supported keys:
            - enable_python_code: bool (default: True)
            - enable_local_rag: bool (default: True)
            - python_code_timeout: int (default: 30)
            - rag_cfg: dict (default: {})
    
    Example:
        tool_config = {
            "enable_python_code": True,
            "enable_local_rag": True,
            "python_code_timeout": 30,
        }
        tool_group = InformalMathToolGroup(tool_config=tool_config)
    """
    def __init__(self, log_requests=True, vllm_cfg=None, verifier_cfg=None, tool_config=None, rag_cfg=None):
        self.log_requests = log_requests

        self.verify_agent = None
        self.python_timeout = 30
        self.enable_python_code = tool_config.get("enable_python_code", True)
        self.enable_local_rag = tool_config.get("enable_local_rag", True)
        self.python_code_timeout = tool_config.get("python_code_timeout", 30)

        # Local RAG configuration
        self.rag_cfg = self._to_dict(tool_config.get("rag_cfg", {}))
        self.enable_python_code = tool_config.get("enable_python_code", True)
        self.enable_local_rag = tool_config.get("enable_local_rag", True)
        self.python_code_timeout = tool_config.get("python_code_timeout", 30)

        # Ground truth for verification (set by environment)
        self.current_ground_truth = None

        if self.log_requests:
            logger.info("InformalMathToolGroup initialized")
            if self.enable_python_code:
                logger.info(f"Python code execution enabled (timeout: {self.python_code_timeout}s)")
            if self.enable_local_rag:
                logger.info("Local RAG enabled")
        super().__init__(name="InformalMathToolGroup")

    def _to_dict(self, cfg):
        """Convert OmegaConf or dict config to plain dict."""
        if cfg is None:
            return {}
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(cfg):
                return OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass
        return dict(cfg) if cfg else {}

    def set_ground_truth(self, ground_truth: str):
        """Set the current ground truth for verification."""
        self.current_ground_truth = ground_truth

    # NOTE: the toolgroup support multiple tools, check ``core.py`` for more details.

    # to simplify the code logic, you MUST align the tool calling tokens with the tool name.
    # for example, if the tool calling token is <informalmath_verify>, the tool name should be informalmath_verify.
    @tool
    def informalmath_verify(self, question: str, solution: str) -> Dict[str, Any]:
        """
        Check if the answer matches the ground truth.
        If no answer is provided, return a score of 0.
        """
        if not question or not solution:
            return {
                "text_result": json.dumps({"score": 0, "text_result": "No question or solution provided."}),
                "score": 0
            }

        try:
            tool_response = call_informalmath_verify(
                question=question,
                solution=solution,
                verify_agent=self.verify_agent,
                ground_truth=self.current_ground_truth,
                enable_python_verify=True,  # Always enabled when ground_truth is available
                python_timeout=self.python_timeout
            )
            score = tool_response.get("score", 0)
            result_text = json.dumps({
                "score": score,
                "text_result": tool_response.get("stdout", "")
            })
            
            if self.log_requests:
                logger.info(f"informalmath_verify: {'Successful' if score else 'Failed'}, got {score} score")
            
            return {"text_result": result_text, "score": score}
            
        except Exception as e:
            error_msg = f"Exception during informalmath_verify: {e}"
            logger.error(error_msg)
            return {
                "text_result": json.dumps({"score": 0, "text_result": error_msg}),
                "score": 0
            }

    @tool
    def python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code locally using subprocess and return the result.
        
        Args:
            code: The Python code to execute.
        
        Returns:
            A dictionary containing:
            - text_result: JSON string with execution results
            - score: 1 if execution successful, 0 otherwise
        """
        if not self.enable_python_code:
            return {
                "text_result": json.dumps({
                    "result": "Python code execution is not enabled.",
                    "status": "disabled"
                }),
                "score": 0
            }
        
        if not code or not code.strip():
            return {
                "text_result": json.dumps({
                    "result": "No code provided.",
                    "status": "error"
                }),
                "score": 0
            }
        
        try:
            # Execute the code
            execution_result = execute_python_code(
                code=code,
                timeout=self.python_code_timeout,
                log_requests=self.log_requests
            )
            
            run_status = execution_result.get("run_status", "Unknown")
            stdout = execution_result.get("stdout", "")
            stderr = execution_result.get("stderr", "")
            returncode = execution_result.get("returncode", -1)
            
            # Format the result
            if run_status == "Finished":
                result_text = json.dumps({
                    "result": stdout,
                    "stderr": stderr,
                    "status": "success",
                    "returncode": returncode
                })
                # logger.debug("python_code tool results: %s", result_text)
                if self.log_requests:
                    logger.info("python_code: Execution successful")
                return {
                    "text_result": result_text,
                    "score": 1
                }
            else:
                # Timeout or Error
                error_msg = stderr if stderr else f"Code execution failed: {run_status}"
                result_text = json.dumps({
                    "result": error_msg,
                    "stderr": stderr,
                    "status": "failed",
                    "returncode": returncode,
                    "run_status": run_status
                })
                # logger.debug("python_code tool results: %s", result_text)
                if self.log_requests:
                    logger.warning(f"python_code: Execution failed - {run_status}")
                return {
                    "text_result": result_text,
                    "score": 0
                }
                
        except Exception as e:
            error_msg = f"Exception during python_code execution: {e}"
            result_text = json.dumps({
                "result": error_msg,
                "status": "error"
            })
            # logger.debug("python_code tool results: %s", result_text)
            logger.error(error_msg)
            return {
                "text_result": result_text,
                "score": 0
            }

    @tool
    def local_rag(self, repo_name: str, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        A search tool system for querying Python package documentation.
        
        Args:
            repo_name: One of [sympy, scipy, numpy, math, cmath, fractions, itertools]
            query: Natural-language query for retrieval, e.g., 
                   'Function interface and examples of calling sympy to solve nonlinear equations'
            top_k: Number of documents to return per sub-query (default: 3)
        
        Returns:
            A dictionary containing:
            - text_result: The query result (code examples, descriptions, usage)
            - score: 1 if successful, 0 otherwise
        """
        if not self.enable_local_rag:
            return {
                "text_result": json.dumps({
                    "result": "Local RAG is not enabled.",
                    "status": "disabled"
                }),
                "score": 0
            }
        
        if not repo_name or not query:
            return {
                "text_result": json.dumps({
                    "result": "Both repo_name and query are required.",
                    "status": "error"
                }),
                "score": 0
            }
        
        try:
            # Get RAG configuration parameters
            rag_base_url = self.rag_cfg.get("rag_base_url")
            chat_base_url = self.rag_cfg.get("chat_base_url")
            chat_model = self.rag_cfg.get("chat_model")
            chat_timeout = self.rag_cfg.get("chat_timeout")
            
            # Execute the RAG query
            result = local_rag_retrieve(
                repo_name=repo_name,
                query=query,
                top_k=top_k,
                rag_base_url=rag_base_url,
                chat_base_url=chat_base_url,
                chat_model=chat_model,
                chat_timeout=chat_timeout,
                log_requests=self.log_requests
            )
            
            status = result.get("status", "error")
            text_result = result.get("text_result", "")
            
            if status == "success":
                result_json = json.dumps({
                    "result": text_result,
                    "status": "success"
                })
                if self.log_requests:
                    logger.info(f"rag_system: Query successful, returned {len(text_result)} chars")
                return {
                    "text_result": result_json,
                    "score": 1
                }
            else:
                error_msg = result.get("error", "Unknown error")
                result_json = json.dumps({
                    "result": text_result,
                    "status": "failed",
                    "error": error_msg
                })
                if self.log_requests:
                    logger.warning(f"rag_system: Query failed - {error_msg}")
                return {
                    "text_result": result_json,
                    "score": 0
                }
                
        except Exception as e:
            error_msg = f"Exception during local_rag query: {e}"
            result_text = json.dumps({
                "result": error_msg,
                "status": "error"
            })
            logger.error(error_msg)
            return {
                "text_result": result_text,
                "score": 0
            }
