# ------------- Evolving prompts -------------
# NOTE: initial prompt for policy agent
# Python code: enabled
# Answer tag: <answer>...</answer>

INFORMAL_MATH_TEMPLATE_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

# NOTE: subsequent prompt for policy agent with previous solutions
INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_NO_TOOL_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Now it's your turn to respond to the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_TOOL_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Now it's your turn to respond to the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

INFORMAL_MATH_TEMPLATE_WITH_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_NO_TOOL_WITH_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_WITH_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_TOOL_WITH_HIS = """You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

INFORMAL_MATH_TEMPLATE_WITH_HIS_FORCE_ANSWER = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_AND_HIS_FORCE_ANSWER = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}. Do NOT include code or your reasoning inside <answer>.
"""

# NOTE: Prompts with local_rag support (Merged: Logic from 2, Python Style from 1)
# Python code: enabled
# Local RAG: enabled
# Answer tag: <answer>...</answer>

INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <local_rag>...</local_rag>: You have access to a RAG System tool to search for documentation or examples (Supported repos: sympy, scipy, numpy, math, cmath, fractions, itertools). Emit exactly ONE <local_rag>...</local_rag> block with a JSON object. Inspect the returned <tool_response> (RAG result). If it disagrees with your reasoning, correct yourself. For example: <local_rag>{{"repo_name": "sympy", "query": "your query here"}}</local_rag>.
3. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_PREVIOUS_SOLUTIONS_NO_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <local_rag>...</local_rag>: You have access to a RAG System tool to search for documentation or examples (Supported repos: sympy, scipy, numpy, math, cmath, fractions, itertools). Emit exactly ONE <local_rag>...</local_rag> block with a JSON object. Inspect the returned <tool_response> (RAG result). If it disagrees with your reasoning, correct yourself. For example: <local_rag>{{"repo_name": "sympy", "query": "your query here"}}</local_rag>.
3. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <local_rag>...</local_rag>: You have access to a RAG System tool to search for documentation or examples (Supported repos: sympy, scipy, numpy, math, cmath, fractions, itertools). Emit exactly ONE <local_rag>...</local_rag> block with a JSON object. Inspect the returned <tool_response> (RAG result). If it disagrees with your reasoning, correct yourself. For example: <local_rag>{{"repo_name": "sympy", "query": "your query here"}}</local_rag>.
3. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_PREVIOUS_SOLUTIONS_WITH_HIS = """
You are a math problem solver agent tasked with solving the given math problem step-by-step.

Your question: {question}

Below are the previous solutions and their verification feedback:
{previous_solutions}

The \\boxed{{1}} within feedback indicates that the previous solution was correct, and \\boxed{{0}} indicates that the previous solution was incorrect. Use the previous solutions and their verification feedback to guide your current step. Remember that if the previous solution was incorrect, you should correct your reasoning and try again.

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <local_rag>...</local_rag>: You have access to a RAG System tool to search for documentation or examples (Supported repos: sympy, scipy, numpy, math, cmath, fractions, itertools). Emit exactly ONE <local_rag>...</local_rag> block with a JSON object. Inspect the returned <tool_response> (RAG result). If it disagrees with your reasoning, correct yourself. For example: <local_rag>{{"repo_name": "sympy", "query": "your query here"}}</local_rag>.
3. <answer>...</answer>: If you are ready to provide the self-contained solution, provide the answer only inside <answer>...</answer>, formatted in LaTeX, e.g., \\boxed{{...}}.
"""

# NOTE: initial evolving prompt for verifier agent
# Python code: enabled
# Answer tag: \\boxed{{1}} or \\boxed{{0}}

VERIFIER_AGENT_TEMPLATE_NO_HIS = """
You are a math verifier agent whose only job is to check whether the policy agent's proposed solution is correct.
Original question:
{question}

Policy agent's latest solution attempt:
{policy_solution}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <report>...</report>: If you are ready to conclude, wrap your verification report inside <report>...</report> tags. The report should:
   - Clearly state whether the policy solution appears correct or incorrect.
   - Explain the key reasoning behind your judgment (keep it concise).
   - End with your judgement in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.
   - The judgement should be enclosed within <report>...</report> tags.
"""

# NOTE: subsequent evolving prompt for verifier agent
# Python code: enabled
# Answer tag: \\boxed{{1}} or \\boxed{{0}}

VERIFIER_AGENT_TEMPLATE_WITH_HIS = """
You are a math verifier agent whose only job is to check whether the policy agent's proposed solution is correct.
Original question:
{question}

Policy agent's latest solution attempt:
{policy_solution}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose ONLY ONE of the following actions (MUST NOT perform both):
1. <python_code>...</python_code>: If computation/checking is helpful, emit exactly ONE <python_code>...</python_code> block with pure Python 3. Inspect the <tool_response> (stdout from your code). If it disagrees with your reasoning, correct yourself.
2. <report>...</report>: If you are ready to conclude, wrap your verification report inside <report>...</report> tags. The report should:
   - Clearly state whether the policy solution appears correct or incorrect.
   - Explain the key reasoning behind your judgment (keep it concise).
   - End with your judgement in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.
   - The judgement should be enclosed within <report>...</report> tags.
"""

VERIFIER_AGENT_TEMPLATE_WITH_HIS_FORCE_REPORT = """
You are a math verifier agent whose only job is to check whether the policy agent's proposed solution is correct.
Original question:
{question}

Policy agent's latest solution attempt:
{policy_solution}

Prior to this step, you have already taken {step_count} step(s).
Below is the interaction history:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, wrap your verification report inside <report>...</report> tags. The report should:
   - Clearly state whether the policy solution appears correct or incorrect.
   - Explain the key reasoning behind your judgment (keep it concise).
   - End with your judgement in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.
   - The judgement should be enclosed within <report>...</report> tags.
"""

# NOTE: initial evolving prompt for verifier agent
# Python code: disabled
# Answer tag: \\boxed{{1}} or \\boxed{{0}}

VERIFIER_AGENT_TEMPLATE_NO_TOOL_NO_HIS = """
You are a math verifier agent whose only job is to check whether the policy agent's proposed solution is correct.
Original question:
{question}

Policy agent's latest solution attempt:
{policy_solution}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, wrap your verification report inside <report>...</report> tags. The report should:
   - Clearly state whether the policy solution appears correct or incorrect.
   - Explain the key reasoning behind your judgment (keep it concise).
   - End with your judgement in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.
   - The judgement should be enclosed within <report>...</report> tags.
"""

# NOTE: subsequent evolving prompt for verifier agent
# Python code: disabled
# Answer tag: \\boxed{{1}} or \\boxed{{0}}

VERIFIER_AGENT_TEMPLATE_NO_TOOL_WITH_HIS = """
You are a math verifier agent whose only job is to check whether the policy agent's proposed solution is correct.
Original question:
{question}

Policy agent's latest solution attempt:
{policy_solution}

Previous policy agent's solutions and their verifications feedback:
{memory_context}

Now it's your turn to respond to the current step.
You should first conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, wrap your verification report inside <report>...</report> tags. The report should:
   - Clearly state whether the policy solution appears correct or incorrect.
   - Explain the key reasoning behind your judgment (keep it concise).
   - End with your judgement in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.
   - The judgement should be enclosed within <report>...</report> tags.
"""

# NOTE: Prompt for aggregating multiple verifier reports into a single representative report
# Used when multiple verifiers produce reports with the same judgment (majority vote)

VERIFIER_REPORT_AGGREGATION_TEMPLATE = """
You are a math verification report aggregator. Multiple verifiers have independently verified a policy agent's solution and reached the same judgment. Your task is to synthesize their reports into a single, comprehensive verification report.

Original question:
{question}

Policy agent's solution:
{policy_solution}

The verifiers have all concluded with judgment: \\boxed{{{majority_judgment}}}

Below are the individual verification reports:
{individual_reports}

Your task:
1. Analyze the key reasoning points from each verifier report.
2. Synthesize the most compelling arguments and evidence into a single, coherent report.
3. Ensure the aggregated report is comprehensive yet concise.
4. Maintain the same final judgment as the individual reports.

Wrap your aggregated verification report inside <report>...</report> tags. The report should:
- Clearly state whether the policy solution is correct or incorrect.
- Combine the strongest reasoning from all verifier reports.
- End with the judgment: \\boxed{{{majority_judgment}}}
"""

SUMMARIZER_TEMPLATE = """### ROLE
You are a Mathematical Logic Auditor. Compress the interaction log into a **Verification Brief**.

### INPUT DATA
The log contains `<think>`, `<python_code>`, `<tool_response>`, and `<answer>` tags.

### PROTOCOL
1. **Filter Noise:** Retain only the mathematical setup, derived constants, and successful logic. Discard syntax errors, backtracking, and internal monologue.
2. **Track Origins:** Explicitly differentiate between values **computed** via code and those **asserted** via intuition or external knowledge.
3. **Format:** Use LaTeX for math. Present as sequential Logical Checkpoints.

### OUTPUT TEMPLATE
**Strategy:** [Brief summary of the approach]

**Logical Checkpoints:**
1. **Setup:** [Variable definitions and initial conditions]
2. **Intermediate Result:** [Key derived values]
3. **Pivotal Step:** [Crucial logic or calculation]
4. **Resolution:** [How the final result was reached]

**Final Claim:** <answer>...</answer>

### TASK
Summarize:
{content}
"""

def get_policy_prompt(enable_python_code=True, use_history=False, use_previous_solutions=False, enable_local_rag=False) -> str:
   """Return the appropriate policy prompt template."""
   if enable_local_rag:
      # Local RAG prompts (with python code support)
      if use_previous_solutions:
         return INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_PREVIOUS_SOLUTIONS_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_PREVIOUS_SOLUTIONS_NO_HIS
      else:
         return INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_WITH_LOCAL_RAG_NO_HIS
   elif enable_python_code:
      if use_previous_solutions:
         return INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_HIS
      else:
         return INFORMAL_MATH_TEMPLATE_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_NO_HIS
   else:
      if use_previous_solutions:
         return INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_TOOL_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_TOOL_NO_HIS
      else:
         return INFORMAL_MATH_TEMPLATE_NO_TOOL_WITH_HIS if use_history else INFORMAL_MATH_TEMPLATE_NO_TOOL_NO_HIS

def get_verifier_prompt(enable_python_code=True, use_history=False) -> str:
   """Return the appropriate verifier prompt template."""
   if enable_python_code:
      return VERIFIER_AGENT_TEMPLATE_WITH_HIS if use_history else VERIFIER_AGENT_TEMPLATE_NO_HIS
   return VERIFIER_AGENT_TEMPLATE_NO_TOOL_WITH_HIS if use_history else VERIFIER_AGENT_TEMPLATE_NO_TOOL_NO_HIS