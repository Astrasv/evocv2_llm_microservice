from typing import Dict, List, Any

def get_affected_cells_analysis_prompt(target_index: int, target_cell_name: str, target_cell_source: str, instruction: str, context_str: str) -> str:
    return f"""Analyze which cells in a DEAP notebook will be affected by this modification.

Target cell (index {target_index}):
Name: {target_cell_name}
Current code:
{target_cell_source}

Modification instruction: {instruction}

{context_str}

12-cell structure:
0: imports, 1: config, 2: creator, 3: evaluate, 4: mate, 5: mutate,
6: select, 7: additional, 8: init, 9: toolbox.register, 10: evolution, 11: results

Determine:
1. Is this a code change (affects dependencies) or just cosmetic (logs, comments)?
2. Which other cells need updates?

Examples:
- "Add logging to mutate" → Only cell 5 (no dependencies)
- "Change mutation to polynomial" → Cells 5 and 9 (registration needs update)
- "Use bounds from config in mutation" → Cells 1, 5, 9 (config, mutate, register)
"""

def get_cell_modification_prompt(cells_str: str, instruction: str, context_str: str) -> str:
    return f"""Modify DEAP notebook cells based on instruction.

Current cells:
{cells_str}

Instruction: {instruction}

{context_str}

Return the modified code for each cell that needs changes.
Preserve the DEAP 12-cell structure and functional programming style.
If the modification requires new packages, provide the full updated list of requirements. Dont just give the newly added modules. Give the entire list of modules requried for the entire notebook
"""
