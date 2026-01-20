from tools import InformalMathToolGroup
tool_group = InformalMathToolGroup(tool_config={"enable_local_rag": True})
result = tool_group.execute_tool('local_rag', {
    'repo_name': 'sympy',
    'query': 'How to solve equations?',
    'top_k': 3
})
print(result['text_result'])