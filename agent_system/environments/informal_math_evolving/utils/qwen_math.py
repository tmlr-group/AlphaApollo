# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen2.5-Math evaluation reward function.
Fully aligned with the official Qwen2.5-Math evaluation suite.
"""
import re
import regex
from math import isclose
from typing import Union, Any
from sympy import simplify, parse_expr

# --- Dependency check ---
try:
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False

def compute_score(solution_str: str, ground_truth: Any) -> float:
    """
    Compute score using Qwen2.5-Math evaluation logic.

    Args:
        solution_str (str): The model's response/solution
        ground_truth (Any): The ground truth answer (typically a string)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    try:
        pred_answer = extract_answer(solution_str)
        is_correct = math_equal(pred_answer, str(ground_truth))

        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"Error in qwen_math evaluation: {e}")
        print(f"  solution_str: {solution_str[:200]}...")
        print(f"  ground_truth: {ground_truth}")
        return 0.0


def _fix_fracs(string):
    """Fix fraction formatting in LaTeX strings."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_sqrt(string):
    """Fix square root formatting in LaTeX strings."""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string: str) -> str:
    """Enhanced string normalization to handle units, LaTeX commands, and other inconsistencies."""
    string = str(string).strip()
    
    # Remove LaTeX wrappers for text/units
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{(.*?)\}", r"\1", string)

    # Normalize and remove common symbols and units
    string = string.replace("^{\\circ}", "")
    string = string.replace("\\circ", "")
    string = string.replace("°", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = re.sub(r"\s*degrees", "", string, flags=re.IGNORECASE)
    
    # Remove common textual units (can be expanded)
    string = re.sub(r"\s*inches\^2", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*square inches", "", string, flags=re.IGNORECASE)

    # Standard replacements
    string = string.replace("\n", "")
    if string.endswith("."):
        string = string[:-1]
    string = string.replace("\\!", "")
    string = string.replace("\\ ", " ")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\n", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # Remove dollar signs
    string = string.replace("$", "")
    
    # Remove spaces and special characters last for clean comparison
    string = re.sub(r" ", "", string)
    string = re.sub("\u200b", "", string)  # Zero-width space

    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    
    return string.strip()


def find_box(string: str) -> str:
    """Find boxed answer in LaTeX format."""
    res = regex.findall(r"\\boxed\{(.*)\}", string)
    if not res:
        res = regex.findall(r"\\fbox\{(.*)\}", string)
    if not res:
        return None
    return res[-1]


def extract_answer(pred_str: str) -> str:
    """Extract answer from prediction string."""
    if pred_str is None:
        return ""
    
    # First try to find boxed answer (Qwen2.5-Math format)
    box_answer = find_box(pred_str)
    if box_answer:
        return box_answer

    tag_answer = extract_answer_segment(pred_str)
    if tag_answer:
        return tag_answer
    
    # Then try to find "#### number" format (GSM8K format)
    gsm8k_answers = re.findall(r"####\s*(-?[0-9\.\,]+)", pred_str)
    if gsm8k_answers:
        return gsm8k_answers[-1].replace(",", "").replace("$", "")
    
    # Then try to find "The answer is X" format
    answer_pattern = re.findall(r'(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)', pred_str)
    if answer_pattern:
        return answer_pattern[-1]
    
    # Last resort: take the last line
    ans_line = pred_str.split('\n')[-1]
    return ans_line


def parse_digits(num):
    """Parse numeric values from strings."""
    num_str = strip_string(str(num))
    num_str = regex.sub(",", "", num_str)
    try:
        return float(num_str)
    except:
        return None


def is_digit(num):
    """Check if a value can be parsed as a number."""
    return parse_digits(num) is not None


def symbolic_equal(a, b):
    """Check if two expressions are symbolically equal."""
    if not LATEX2SYMPY_AVAILABLE:
        try:
            return simplify(a) == simplify(b)
        except:
            return False

    def _parse(s):
        try:
            return latex2sympy(s)
        except:
            try:
                return parse_expr(s)
            except:
                return s
    
    try:
        if simplify(_parse(a) - _parse(b)) == 0:
            return True
    except:
        pass
    return False


def math_equal(prediction: Union[bool, float, str], reference: Union[float, str]) -> bool:
    """
    Check if prediction matches reference using robust math comparison.
    Handles numerical, symbolic, and textual comparisons.
    """
    if prediction is None or reference is None:
        return False
    
    # Keep raw strings for special checks like \pm
    pred_raw = str(prediction)
    ref_raw = str(reference)

    # Path for \pm: check if one string has \pm and the other is a list of 2
    pm_pattern = r"(.+?)\s*\\pm\s*(.+)"
    
    def check_pm_match(pm_str, list_str):
        pm_match = re.match(pm_pattern, pm_str.strip())
        if not pm_match:
            return False
        
        parts = re.split(r'[,;]', list_str)
        if len(parts) != 2:
            return False
        
        A = pm_match.group(1).strip()
        B = pm_match.group(2).strip()
        
        # Create sets of normalized strings for order-agnostic comparison
        pm_set = {strip_string(f"{A}+{B}"), strip_string(f"{A}-{B}")}
        list_set = {strip_string(parts[0]), strip_string(parts[1])}
        
        return pm_set == list_set

    if r"\pm" in ref_raw and check_pm_match(ref_raw, pred_raw):
        return True
    if r"\pm" in pred_raw and check_pm_match(pred_raw, ref_raw):
        return True

    # Apply enhanced normalization to both prediction and reference
    prediction_str = strip_string(pred_raw)
    reference_str = strip_string(ref_raw)
    
    # Path 0: Direct string equality after normalization
    if prediction_str == reference_str:
        return True

    # Path 1: Numerical comparison
    if is_digit(prediction_str) and is_digit(reference_str):
        pred_float = parse_digits(prediction_str)
        ref_float = parse_digits(reference_str)
        if pred_float is not None and ref_float is not None and isclose(pred_float, ref_float, rel_tol=1e-4):
            return True

    # Path 2: Unordered Tuple/Vector comparison
    pred_parts = [p.strip() for p in re.split(r'[,;]', prediction_str.strip('()[]{}')) if p.strip()]
    ref_parts = [r.strip() for r in re.split(r'[,;]', reference_str.strip('()[]{}')) if r.strip()]
    if len(pred_parts) > 1 and len(pred_parts) == len(ref_parts):
        # Sort the parts before comparing to handle unordered lists
        if sorted(pred_parts) == sorted(ref_parts):
            return True

    # Path 3: Symbolic comparison
    if symbolic_equal(prediction_str, reference_str):
        return True

    return False

def extract_answer_segment(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()