# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
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

from typing import List, Tuple, Optional
import re
from agent_system.environments.prompts import (
    VERIFIER_AGENT_TEMPLATE_NO_HIS,
    VERIFIER_AGENT_TEMPLATE_WITH_HIS,
)

# add new tool calling tokens here
TOOL_CALLING_TOKENS = [
    "python_code",
    "local_rag",
    "rag_search",
    "web_search",
    "informalmath_verify",
]

def _postprocess_action(action: str) -> str:
    """Trim everything *after* the *first* closing `</answer>` or `</TOOL_CALLING_TOKEN>` tag.

    Answer has the highest priority. This guards against a common LLM hallucination
    where an action contains several concatenated XML-like snippets. By hard-cutting
    at the first relevant close tag we can safely apply non-greedy regex below.
    """
    
    # Check for </answer> tag first (highest priority)
    answer_pos = action.find("</answer>")
    if answer_pos != -1:
        return action[:answer_pos] + "</answer>"
    
    # Find the earliest tool calling closing tag
    earliest_pos = len(action)
    earliest_tag = None
    
    for tool_calling_token in TOOL_CALLING_TOKENS:
        closing_tag = f"</{tool_calling_token}>"
        pos = action.find(closing_tag)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            earliest_tag = closing_tag
    
    # If we found a closing tag, trim at it
    if earliest_tag:
        return action[:earliest_pos] + earliest_tag
    
    return action


def informal_math_training_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project a list of LLM *actions* into (`results`, `valids`).

    Extraction logic (order matters):
        1. Grab the **first** complete ``<TOOL_CALLING_TOKEN>…</TOOL_CALLING_TOKEN>`` block (case‑insensitive).
        2. If absent, grab the **first** complete ``<answer>…</answer>`` block.
        3. If still absent, store an empty string.

    Validity logic (independent of extraction): ``valids[i]`` flips to **0** when
    the *original* action text satisfies any of:
        1. Contains **both** ``<TOOL_CALLING_TOKEN>`` and ``<answer>`` tags.
        2. Contains more than one ``<TOOL_CALLING_TOKEN>`` tag or more than one ``<answer>`` tag.

    The extracted block (if any) is **not** cleared when a validity rule fails –
    downstream callers can still inspect the fragment while trusting the flag.
    """

    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    # --- Pre‑compiled patterns ------------------------------------------------
    re_tool_calling_blocks = {}
    re_tool_calling_tags = {}
    for tool_calling_token in TOOL_CALLING_TOKENS:
        re_tool_calling_blocks[tool_calling_token] = re.compile(f"<{tool_calling_token}>(.*?)</{tool_calling_token}>", re.IGNORECASE | re.DOTALL)
        re_tool_calling_tags[tool_calling_token] = re.compile(f"<{tool_calling_token}>", re.IGNORECASE)
    
    re_answer_block = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)

    for i, action in enumerate(actions):
        original_action = action  # Keep untouched for validity checks
        trimmed_action = _postprocess_action(action)
        # --- Extraction -----------------------------------------------------
        # Answer has the highest priority - check for <answer> block first
        m = re_answer_block.search(trimmed_action)
        if m:
            results.append(f"<answer>{m.group(1).strip()}</answer>")
        else:
            # If no answer block is found, check for tool calling blocks
            found = False
            for tool_calling_token in TOOL_CALLING_TOKENS:
                m = re_tool_calling_blocks[tool_calling_token].search(trimmed_action)
                if m:
                    results.append(f"<{tool_calling_token}>{m.group(1).strip()}</{tool_calling_token}>")
                    found = True
                    break
            if not found:
                results.append("")
                valids[i] = 0

        # --- Validity checks -------------------------------------------------
        n_tool_calling = sum(len(re_tool_calling_tags[token].findall(original_action)) for token in TOOL_CALLING_TOKENS)
        n_answer = len(re_answer_tag.findall(original_action))

        # Both tool calling and answer present
        if n_tool_calling and n_answer:
            valids[i] = 0
            continue
        # Multiple identical tags
        if n_tool_calling > 1 or n_answer > 1:
            valids[i] = 0

    return results, valids

def _test_projection():
    '''
    results:
    [
        "<python_code>print(1+1)</python_code>",
        "<rag_search>What is the capital of France?</rag_search>",
        "<web_search>What is the capital of France?</web_search>",
        "<answer>2</answer>",
        "<web_search>prime numbers</web_search>",
        "<answer>0</answer>",
    ]
    valids: [1, 1, 1, 0, 0, 0]
    '''
    actions = [
        "Some random text<python_code>print(1+1)</python_code> Some random text <tool_response>2</tool_response>",
        "<rag_search>What is the capital of France?</rag_search><tool_response>Paris</tool_response>",
        "<web_search>What is the capital of France?</web_search><tool_response>Paris</tool_response>",
        "<python_code>print('hello')</python_code><answer>2</answer>",  # answer has priority - trims and extracts <answer>
        "<web_search>prime numbers</web_search>Some random text<python_code>is_prime(7)</python_code>",  # first tool is extracted after trimming
        "Some random text<informalmath_>8</informalmath_>Some random text<answer>0</answer>Some random text",  # answer has priority - trims and extracts <answer>
    ]
    results, valids = informal_math_training_projection(actions)
    print(f"results: {results}")
    print(f"valids: {valids}")

if __name__ == "__main__":
    _test_projection()