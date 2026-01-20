import re
import json
import logging
from typing import Dict, Any, Optional
from agent_system.environments.prompts import *
from tools.python_code import execute_python_code

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _agent_generate(verify_agent, prompt: str) -> str:
    """
    Generate verification response from agent.

    Args:
        verify_agent: Agent to generate response
        prompt: Verification prompt

    Returns:
        Agent response
    """
    return verify_agent.get_action_from_gpt(prompt)


def _extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from solution text.

    Args:
        text: Solution text that may contain <python_code>...</python_code> tags

    Returns:
        Extracted Python code, or None if not found
    """
    pattern = r"<python_code>(.*?)</python_code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _execute_python_and_verify(solution: str, ground_truth: str, python_timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code from solution and verify against ground truth.

    Args:
        solution: The complete solution text
        ground_truth: Ground truth answer
        python_timeout: Timeout for Python execution in seconds

    Returns:
        Dict containing execution and verification results
    """
    result = {
        "has_code": False,
        "executed": False,
        "python_result": None,
        "python_status": "no_code",
        "matches_gt": None
    }

    python_code = _extract_python_code(solution)
    if not python_code:
        return result

    result["has_code"] = True

    exec_result = execute_python_code(code=python_code, timeout=python_timeout, log_requests=False)
    result["executed"] = True
    result["python_status"] = exec_result["run_status"]

    if exec_result["run_status"] == "Finished" and exec_result["returncode"] == 0:
        stdout = exec_result["stdout"].strip()
        result["python_result"] = stdout

        # Simple containment check
        gt_normalized = str(ground_truth).strip().lower()
        result_normalized = stdout.lower()
        result["matches_gt"] = gt_normalized in result_normalized
    else:
        result["python_result"] = exec_result.get("stderr", "Execution failed")
        result["matches_gt"] = False

    return result


def _build_judge_prompt(question: str, solution: str, python_result: Optional[Dict[str, Any]] = None) -> str:
    """
    Build verification prompt with optional Python execution results.

    Args:
        question: The math problem
        solution: The solution to verify
        python_result: Optional Python execution result

    Returns:
        Formatted prompt
    """
    parts = [VERIFIER_PROMPT, f"Problem: {question}", f"Solution: {solution}"]

    if python_result and python_result.get("executed"):
        parts.append(f"\nPython execution status: {python_result['python_status']}")
        if python_result.get("python_result"):
            parts.append(f"Python output: {python_result['python_result']}")
        if python_result.get("matches_gt") is not None:
            parts.append(f"Matches ground truth: {python_result['matches_gt']}\n")

    return "".join(parts)


def _parse_judgement(text: str) -> float:
    """
    Parse judgement from verifier response.

    Args:
        text: Verifier response text

    Returns:
        Score (1.0 for correct, 0.0 for incorrect)
    """
    m = re.search(r"\\boxed\{([01])\}", text)
    if not m:
        # Try a more lenient fallback
        m = re.search(r"boxed\{([01])\}", text)
    if m and m.group(1) == "1":
        return 1.0
    return 0.0


def call_informalmath_verify(
    question: str,
    solution: str,
    verify_agent,
    ground_truth: Optional[str] = None,
    enable_python_verify: bool = True,
    python_timeout: int = 30
) -> Dict[str, Any]:
    """
    LLM-as-a-judge verification with optional Python code execution.

    Args:
        question: The math problem
        solution: Model-produced solution text
        verify_agent: Agent to generate verification judgement
        ground_truth: Optional ground truth answer (for Python verification)
        enable_python_verify: Whether to execute Python code from solution
        python_timeout: Timeout for Python execution in seconds

    Returns:
        Dict with score (0.0 or 1.0), stdout, stderr, and optional python_verification
    """
    python_result = None

    if enable_python_verify and ground_truth:
        python_result = _execute_python_and_verify(solution, ground_truth, python_timeout)
        if logger.isEnabledFor(logging.INFO) and python_result.get("executed"):
            logger.info(f"Python verification: status={python_result['python_status']}, "
                       f"matches_gt={python_result.get('matches_gt')}")

    user_prompt = _build_judge_prompt(question, solution, python_result)
    content = _agent_generate(verify_agent, user_prompt)
    print(f"==> DEBUG: Verifier response: {content}")
    score = _parse_judgement(content)
    print(f"==> DEBUG: Verifier score: {score}")
    stdout_parts = [f"informalmath_verify: score={score}\n"]

    if python_result and python_result.get("executed"):
        stdout_parts.append(f"\n[Python Verification]")
        stdout_parts.append(f"Status: {python_result['python_status']}")
        if python_result.get("python_result"):
            result_str = str(python_result['python_result'])
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            stdout_parts.append(f"Result: {result_str}")
        stdout_parts.append(f"Matches GT: {python_result.get('matches_gt')}\n")

    stdout_parts.append(f"\n[Prompt]\n{user_prompt}\n")
    stdout_parts.append(f"\n[Judge Response]\n{content}")

    stdout = "\n".join(stdout_parts)

    if logger.isEnabledFor(logging.INFO):
        logger.info(f"informalmath_verify -> score={score}")

    result = {"score": score, "stdout": stdout, "stderr": ""}

    if python_result:
        result["python_verification"] = python_result

    return result
