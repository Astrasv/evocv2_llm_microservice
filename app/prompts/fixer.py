def get_fix_analysis_prompt(traceback: str, context_str: str, notebook_summary: str, additional_context: str) -> str:
    return f"""Analyze and fix this DEAP notebook error.

Error traceback:
{traceback}

{context_str}

Current notebook structure (12 cells):
{notebook_summary}

Additional context: {additional_context or 'None'}

Common DEAP notebook errors:
1. toolbox.register() called before function definitions
2. Missing return tuple comma in evaluate/mutate functions
3. Incorrect creator.create() usage
4. Variables used before definition
5. Missing imports

Analyze the error and provide a fix plan.
Return structured fixes for the affected cells.
If the fix requires new packages, provide the full updated list of requirements.
"""
