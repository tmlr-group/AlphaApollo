# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Optional
import ast
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 3

PRE_IMPORT_LIBS = """
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""

def check_forbidden_imports(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True

    # check forbidden imports
    forbidden_modules = {
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                if name.name.split('.')[0] in forbidden_modules:
                    return True

    # check forbidden input()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "input":
                return True

    return False

# a very simple version of indentation fixing
def fix_indentation(code_lines):
    fixed_lines = []
    # Make a mutable copy of code_lines for fixing
    code_lines = list(code_lines)
    for i, line in enumerate(code_lines):
        curr_strip = line.lstrip()
        curr_indent = len(line) - len(curr_strip)
        # If the line is empty (only newline or all whitespace), add it directly
        if curr_strip == "":
            fixed_lines.append(line)
            continue
        # If the line is a comment, add it directly
        if curr_strip.startswith("#"):
            fixed_lines.append(line)
            continue
        if i == 0:
            fixed_lines.append(line.lstrip()) # No indentation for the first line
            code_lines[i] = line.lstrip()
            continue
        prev_line = code_lines[i-1]
        prev_strip = prev_line.lstrip()
        prev_indent = len(prev_line) - len(prev_strip)
        # If the previous line is a comment or empty, look upwards for the nearest non-comment, non-empty line
        j = i - 1
        while j >= 0 and (code_lines[j].lstrip().startswith("#") or code_lines[j].lstrip() == ""):
            j -= 1
        if j >= 0:
            prev_line_non_comment = code_lines[j]
            prev_strip_non_comment = prev_line_non_comment.lstrip()
            prev_indent_non_comment = len(prev_line_non_comment) - len(prev_strip_non_comment)
        else:
            prev_line_non_comment = ""
            prev_indent_non_comment = 0
        # If the previous non-comment, non-empty line does not end with a colon,
        # and the current line is more indented than it, remove the extra indentation
        if (':' not in prev_line_non_comment.rstrip()) and (curr_indent > prev_indent_non_comment):
            new_line = ' ' * prev_indent_non_comment + curr_strip
            fixed_lines.append(new_line)
            code_lines[i] = new_line  # Update the original code_lines for further fixing
        else:
            fixed_lines.append(line)
            code_lines[i] = line  # Keep in sync
    return fixed_lines

def wrap_python_code(code: str) -> str:
    """
    Wrap Python code with print statements.
    """
    lines = code.rstrip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if not last_line.startswith('print('):
            try:
                ast.parse(last_line, mode='eval')
                lines[-1] = f'print({last_line})'
            except Exception:
                try:
                    tree = ast.parse(code)
                    last_assign = None
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            last_assign = node
                    if last_assign and isinstance(last_assign.targets[0], ast.Name):
                        var_name = last_assign.targets[0].id
                        lines.append(f'print({var_name})')
                except Exception:
                    pass
    lines = fix_indentation(lines)
    wrapper_code = '\n'.join(lines)

    # Safely escape newlines within f-strings
    pattern = re.compile(r"f([\"']{1,3})(.*?)(\1)", re.DOTALL)
    def replacer(match):
        quote = match.group(1)
        content = match.group(2)
        content = content.replace('\n', '\\n')
        return f"f{quote}{content}{quote}"
    wrapper_code = pattern.sub(replacer, wrapper_code)

    return PRE_IMPORT_LIBS + "\n" + wrapper_code

def execute_python_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
) -> Dict[str, Any]:
    """
    Execute Python code locally using subprocess.
    
    Args:
        code: The Python code to execute.
        timeout: The timeout for code execution in seconds.
        log_requests: Whether to log execution details.
    
    Returns:
        A dictionary containing execution results:
        {
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "run_status": str  # "Finished", "Timeout", or "Error"
        }
    """
    if not code or not code.strip():
        return {
            "stdout": "",
            "stderr": "No code provided.",
            "returncode": -1,
            "run_status": "Error"
        }
    
    if log_requests:
        logger.info(f"Executing Python code locally (timeout: {timeout}s)")
        logger.debug(f"Code to execute:\n{code[:200]}...")  # Log first 200 chars
    
    if check_forbidden_imports(code):
        return {
            "stdout": "",
            "stderr": "Forbidden imports or input() found.",
            "returncode": -1,
            "run_status": "Error"
        }
    
    temp_file = None
    code = wrap_python_code(code)
    try:
        # Create a temporary file for the code
        fd, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code_", text=True)
        
        # Write code to file
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute the code using subprocess
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None,  # Use current working directory
                env=os.environ.copy(),  # Use current environment
            )
            
            run_status = "Finished" if result.returncode == 0 else "Error"
            
            if log_requests:
                if result.returncode == 0:
                    logger.info(f"Code execution successful (returncode: {result.returncode})")
                else:
                    logger.warning(f"Code execution failed (returncode: {result.returncode})")
                    if result.stderr:
                        logger.debug(f"Stderr: {result.stderr[:500]}")
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "run_status": run_status
            }
            
        except subprocess.TimeoutExpired:
            if log_requests:
                logger.warning(f"Code execution timed out after {timeout} seconds")
            return {
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout} seconds",
                "returncode": -1,
                "run_status": "Timeout"
            }
        except Exception as e:
            if log_requests:
                logger.error(f"Error executing code: {e}")
            return {
                "stdout": "",
                "stderr": f"Exception during code execution: {str(e)}",
                "returncode": -1,
                "run_status": "Error"
            }
            
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                if log_requests:
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                if log_requests:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")


def test_python_code_executor():
    """
    Test function to verify the Python code executor is working correctly.
    Tests various scenarios including successful execution, errors, timeouts, and edge cases.
    """
    print("=" * 60)
    print("Testing Python Code Executor")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic successful execution
    print("\n[Test 1] Basic successful execution")
    print("-" * 60)
    code1 = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
    result1 = execute_python_code(code1, timeout=5, log_requests=False)
    success1 = result1["run_status"] == "Finished" and result1["returncode"] == 0
    test_results.append(("Basic execution", success1))
    print(f"Status: {result1['run_status']}")
    print(f"Return code: {result1['returncode']}")
    print(f"Output: {result1['stdout'].strip()}")
    print(f"✓ PASS" if success1 else "✗ FAIL")
    
    # Test 2: Code with syntax error
    print("\n[Test 2] Code with syntax error")
    print("-" * 60)
    code2 = """
print("Hello"
# Missing closing parenthesis
"""
    result2 = execute_python_code(code2, timeout=5, log_requests=False)
    success2 = result2["run_status"] == "Error" and result2["returncode"] != 0
    test_results.append(("Syntax error handling", success2))
    print(f"Status: {result2['run_status']}")
    print(f"Return code: {result2['returncode']}")
    print(f"Error output: {result2['stderr'][:100]}...")
    print(f"✓ PASS" if success2 else "✗ FAIL")
    
    # Test 3: Code with runtime error
    print("\n[Test 3] Code with runtime error")
    print("-" * 60)
    code3 = """
x = 10
y = 0
result = x / y  # Division by zero
print(result)
"""
    result3 = execute_python_code(code3, timeout=5, log_requests=False)
    success3 = result3["run_status"] == "Error" and result3["returncode"] != 0
    test_results.append(("Runtime error handling", success3))
    print(f"Status: {result3['run_status']}")
    print(f"Return code: {result3['returncode']}")
    print(f"Error output: {result3['stderr']}...")
    print(f"✓ PASS" if success3 else "✗ FAIL")
    
    # Test 4: Empty code
    print("\n[Test 4] Empty code")
    print("-" * 60)
    code4 = ""
    result4 = execute_python_code(code4, timeout=5, log_requests=False)
    success4 = result4["run_status"] == "Error"
    test_results.append(("Empty code handling", success4))
    print(f"Status: {result4['run_status']}")
    print(f"Error message: {result4['stderr']}")
    print(f"✓ PASS" if success4 else "✗ FAIL")
    
    # Test 5: Code with output
    print("\n[Test 5] Code with multiple outputs")
    print("-" * 60)
    code5 = """
for i in range(3):
    print(f"Line {i+1}")
print("Done!")
"""
    result5 = execute_python_code(code5, timeout=5, log_requests=False)
    success5 = result5["run_status"] == "Finished" and "Line 1" in result5["stdout"]
    test_results.append(("Multiple outputs", success5))
    print(f"Status: {result5['run_status']}")
    print(f"Output:\n{result5['stdout']}")
    print(f"✓ PASS" if success5 else "✗ FAIL")
    
    # Test 6: Mathematical computation
    print("\n[Test 6] Mathematical computation")
    print("-" * 60)
    code6 = """
import math
a = 5
b = 12
c = math.sqrt(a**2 + b**2)
print(f"Hypotenuse: {c}")
"""
    result6 = execute_python_code(code6, timeout=5, log_requests=False)
    success6 = result6["run_status"] == "Finished" and "13.0" in result6["stdout"]
    test_results.append(("Mathematical computation", success6))
    print(f"Status: {result6['run_status']}")
    print(f"Output: {result6['stdout'].strip()}")
    print(f"✓ PASS" if success6 else "✗ FAIL")
    
    # Test 7: Code with imports
    print("\n[Test 7] Code with standard library imports")
    print("-" * 60)
    code7 = """
import json
import datetime

data = {"date": str(datetime.datetime.now()), "value": 42}
json_str = json.dumps(data)
print(json_str)
"""
    result7 = execute_python_code(code7, timeout=5, log_requests=False)
    success7 = result7["run_status"] == "Finished" and "value" in result7["stdout"] and "42" in result7["stdout"]
    test_results.append(("Standard library imports", success7))
    print(f"Status: {result7['run_status']}")
    print(f"Output: {result7['stdout'].strip()[:100]}...")
    print(f"✓ PASS" if success7 else "✗ FAIL")
    
    # Test 8: Timeout test (optional, may take time)
    print("\n[Test 8] Timeout handling (2 second timeout)")
    print("-" * 60)
    code8 = """
import time
time.sleep(5)  # Sleep for 5 seconds
print("This should not print")
"""
    result8 = execute_python_code(code8, timeout=2, log_requests=False)
    success8 = result8["run_status"] == "Timeout"
    test_results.append(("Timeout handling", success8))
    print(f"Status: {result8['run_status']}")
    print(f"Error: {result8['stderr'][:100]}...")
    print(f"✓ PASS" if success8 else "✗ FAIL")
    # Test 9: Input handling
    print("\n[Test 9] Input handling")
    print("-" * 60)
    code9 = """
import time
input("Enter your name: ")
print("Hello, " + name)
"""
    result9 = execute_python_code(code9, timeout=2, log_requests=False)
    success9 = result9["run_status"] == "Error"
    test_results.append(("Input handling", success9))
    print(f"Status: {result9['run_status']}")
    print(f"Output: {result9['stdout']}")
    print(f"Error: {result9['stderr']}")
    print(f"✓ PASS" if success9 else "✗ FAIL")
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Python code executor is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = test_python_code_executor()
    sys.exit(0 if success else 1)
