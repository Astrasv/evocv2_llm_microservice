"""Modifier agent - Modifies existing notebooks via natural language with Mem0 integration."""

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings
from app.models import ModifyRequest, NotebookStructure, NotebookCell
from app.memory import enhanced_memory
from app.utils import format_code
import logging

logger = logging.getLogger(__name__)


class AffectedCellsAnalysis(BaseModel):
    """LLM analysis of which cells are affected by a modification."""
    target_cell_index: int = Field(description="Primary cell to modify")
    affected_cells: List[int] = Field(description="Additional cells that need updates")
    reasoning: str = Field(description="Why these cells are affected")


class CellModification(BaseModel):
    """Single cell modification."""
    cell_index: int
    new_code: str
    change_description: str


class ModificationResult(BaseModel):
    """Result of cell modification."""
    modifications: List[CellModification]
    changes_summary: List[str]
    requirements: Optional[str] = Field(None, description="Updated newline-separated requirements (if changed)")


class NotebookModifier:
    """
    Modifies existing DEAP notebooks using 3-case architecture:
    1. Targeted cell modification (when cell_name provided)
    2. Notebook-level modification (no cell_name)
    3. Error fixing is handled by fixer.py
    """

    # Map cell names to indices
    CELL_MAP = {
        "imports": 0,
        "import": 0,
        "config": 1,
        "configuration": 1,
        "creator": 2,
        "evaluate": 3,
        "evaluation": 3,
        "fitness": 3,
        "mate": 4,
        "crossover": 4,
        "mutate": 5,
        "mutation": 5,
        "select": 6,
        "selection": 6,
        "additional": 7,
        "operators": 7,
        "init": 8,
        "initialization": 8,
        "register": 9,
        "toolbox": 9,
        "evolution": 10,
        "loop": 10,
        "algorithm": 10,
        "results": 11,
        "plots": 11,
        "plotting": 11
    }

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )

    def modify(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        Modify notebook based on request type.

        Returns:
            - Modified notebook
            - List of change descriptions
            - List of modified cell indices
        """
        logger.info(f"Modifying notebook {request.notebook_id} for user {request.user_id}")
        logger.info(f"Instruction: {request.instruction}")
        logger.info(f"Cell name: {request.cell_name}")

        if request.cell_name:
            # Case 1: Targeted cell modification
            return self._targeted_modification(request)
        else:
            # Case 2: Notebook-level modification
            return self._notebook_level_modification(request)

    # targetted modification

    def _targeted_modification(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        Modify a specific cell with minimal context.

        Steps:
        1. Get target cell index from cell_name
        2. Query Mem0 for user preferences and learned dependencies
        3. LLM determines affected cells (dynamic dependency analysis)
        4. Fetch only required cells
        5. LLM modifies those cells
        6. Store patterns in Mem0
        """
        logger.info(f"Targeted modification for cell: {request.cell_name}")

        #  Get target cell index
        target_index = self._get_cell_index(request.cell_name)
        if target_index is None:
            logger.error(f"Invalid cell name: {request.cell_name}")
            # Fallback to notebook-level
            return self._notebook_level_modification(request)

        #  Get Mem0 context
        mem0_context = self._get_mem0_context_for_cell(
            request.user_id,
            request.notebook_id,
            request.cell_name
        )

        #  LLM determines affected cells
        affected_analysis = self._analyze_affected_cells(
            request,
            target_index,
            mem0_context
        )

        all_cell_indices = sorted(set([affected_analysis.target_cell_index] + affected_analysis.affected_cells))
        logger.info(f"Cells to modify: {all_cell_indices}")

        #  Get cells for modification
        cells_for_llm = [
            {
                "index": i,
                "name": request.notebook.cells[i].cell_name or f"cell_{i}",
                "code": request.notebook.cells[i].source
            }
            for i in all_cell_indices
        ]

        #  Modify cells with LLM
        modification_result = self._modify_cells_with_llm(
            cells_for_llm,
            request.instruction,
            mem0_context,
            target_cell_name=request.cell_name
        )

        #  Apply modifications to notebook
        modified_notebook = self._apply_cell_modifications(
            request.notebook,
            modification_result.modifications
        )
        
        if modification_result.requirements:
            modified_notebook.requirements = modification_result.requirements

        #  Store in Mem0
        self._store_targeted_modification_in_mem0(
            request,
            target_index,
            all_cell_indices,
            modification_result
        )

        return modified_notebook, modification_result.changes_summary, all_cell_indices

    def _get_cell_index(self, cell_name: str) -> Optional[int]:
        """Convert cell name to index."""
        normalized = cell_name.lower().strip()
        return self.CELL_MAP.get(normalized)

    def _get_mem0_context_for_cell(
        self,
        user_id: str,
        notebook_id: str,
        cell_name: str
    ) -> Dict[str, Any]:
        """Get relevant Mem0 context for cell modification."""
        context = {
            "user_preferences": [],
            "cell_patterns": [],
            "learned_dependencies": [],
            "notebook_history": []
        }

        try:
            # User preferences for this cell type
            prefs = enhanced_memory.search_user_context(
                user_id=user_id,
                query=f"preferences for {cell_name}",
                limit=3
            )
            context["user_preferences"] = [p.get("memory", p.get("content", "")) for p in prefs]

            # Past modifications to this cell type
            patterns = enhanced_memory.get_cell_patterns(
                user_id=user_id,
                cell_name=cell_name,
                limit=3
            )
            context["cell_patterns"] = [p.get("memory", p.get("content", "")) for p in patterns]

            # Learned dependencies
            deps = enhanced_memory.get_learned_dependencies(
                user_id=user_id,
                cell_name=cell_name
            )
            context["learned_dependencies"] = deps

            # Notebook history
            history = enhanced_memory.get_notebook_history(
                user_id=user_id,
                notebook_id=notebook_id,
                limit=3
            )
            context["notebook_history"] = [h.get("memory", h.get("content", "")) for h in history]

        except Exception as e:
            logger.error(f"Error getting Mem0 context: {e}")

        return context

    def _analyze_affected_cells(
        self,
        request: ModifyRequest,
        target_index: int,
        mem0_context: Dict[str, Any]
    ) -> AffectedCellsAnalysis:
        """Use LLM to determine which cells are affected."""
        target_cell = request.notebook.cells[target_index]

        # Build context string
        context_str = self._format_mem0_context(mem0_context)

        prompt = f"""Analyze which cells in a DEAP notebook will be affected by this modification.

Target cell (index {target_index}):
Name: {target_cell.cell_name}
Current code:
{target_cell.source}

Modification instruction: {request.instruction}

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

        try:
            analysis = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=AffectedCellsAnalysis,
                max_tokens=500,
                temperature=0.1
            )
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze affected cells: {e}")
            # Fallback: assume only target cell
            return AffectedCellsAnalysis(
                target_cell_index=target_index,
                affected_cells=[],
                reasoning="Fallback: only modifying target cell"
            )

    def _modify_cells_with_llm(
        self,
        cells: List[Dict],
        instruction: str,
        mem0_context: Dict[str, Any],
        target_cell_name: Optional[str] = None
    ) -> ModificationResult:
        """Use LLM to modify cells."""
        context_str = self._format_mem0_context(mem0_context)

        cells_str = "\n\n".join([
            f"Cell {c['index']} ({c['name']}):\n{c['code']}"
            for c in cells
        ])

        prompt = f"""Modify DEAP notebook cells based on instruction.

Current cells:
{cells_str}

Instruction: {instruction}

{context_str}

Return the modified code for each cell that needs changes.
Preserve the DEAP 12-cell structure and functional programming style.
If the modification requires new packages, provide the full updated list of requirements. Dont just give the newly added modules. Give the entire list of modules requried for the entire notebook
"""

        try:
            result = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=ModificationResult,
                max_tokens=3000,
                temperature=0.3
            )
            return result
        except Exception as e:
            logger.error(f"Failed to modify cells with LLM: {e}")
            # Fallback: return empty modifications
            return ModificationResult(
                modifications=[],
                changes_summary=[f"Error: {str(e)}"]
            )

    #  notebook-level modification

    def _notebook_level_modification(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        Modify entire notebook for generic/complex changes.

        Uses full notebook context + Mem0 for holistic modifications.
        """
        logger.info("Notebook-level modification (full context)")

        # Get Mem0 context
        mem0_context = self._get_mem0_context_for_notebook(
            request.user_id,
            request.notebook_id
        )

        # Build full notebook context
        all_cells = [
            {
                "index": i,
                "name": cell.cell_name or f"cell_{i}",
                "code": cell.source
            }
            for i, cell in enumerate(request.notebook.cells)
        ]

        # Modify with full context
        modification_result = self._modify_cells_with_llm(
            all_cells,
            request.instruction,
            mem0_context,
            target_cell_name=None
        )

        # Apply modifications
        modified_notebook = self._apply_cell_modifications(
            request.notebook,
            modification_result.modifications
        )
        
        if modification_result.requirements:
            modified_notebook.requirements = modification_result.requirements

        # Extract modified cell indices
        modified_indices = [m.cell_index for m in modification_result.modifications]

        # Store in Mem0
        self._store_notebook_modification_in_mem0(
            request,
            modified_indices,
            modification_result
        )

        return modified_notebook, modification_result.changes_summary, modified_indices

    def _get_mem0_context_for_notebook(
        self,
        user_id: str,
        notebook_id: str
    ) -> Dict[str, Any]:
        """Get Mem0 context for notebook-level modifications."""
        context = {
            "user_preferences": [],
            "notebook_history": [],
            "common_patterns": []
        }

        try:
            # Overall user preferences
            prefs = enhanced_memory.search_user_context(
                user_id=user_id,
                query="optimization preferences and patterns",
                limit=5
            )
            context["user_preferences"] = [p.get("memory", p.get("content", "")) for p in prefs]

            # Notebook history
            history = enhanced_memory.get_notebook_history(
                user_id=user_id,
                notebook_id=notebook_id,
                limit=5
            )
            context["notebook_history"] = [h.get("memory", h.get("content", "")) for h in history]

        except Exception as e:
            logger.error(f"Error getting notebook Mem0 context: {e}")

        return context

    # helpers

    def _format_mem0_context(self, context: Dict[str, Any]) -> str:
        """Format Mem0 context for LLM prompt."""
        parts = []

        if context.get("user_preferences"):
            parts.append("User preferences:\n- " + "\n- ".join(context["user_preferences"]))

        if context.get("cell_patterns"):
            parts.append("Past cell modifications:\n- " + "\n- ".join(context["cell_patterns"]))

        if context.get("learned_dependencies"):
            deps = ", ".join(context["learned_dependencies"])
            parts.append(f"Learned dependencies: {deps}")

        if context.get("notebook_history"):
            parts.append("Recent notebook changes:\n- " + "\n- ".join(context["notebook_history"][:2]))

        return "\n\n".join(parts) if parts else "No prior context available."

    def _apply_cell_modifications(
        self,
        notebook: NotebookStructure,
        modifications: List[CellModification]
    ) -> NotebookStructure:
        """Apply LLM modifications to notebook."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        for mod in modifications:
            if 0 <= mod.cell_index < 12:
                cells[mod.cell_index] = NotebookCell(
                    cell_type="code",
                    cell_name=cells[mod.cell_index].cell_name,
                    source=format_code(mod.new_code),
                    execution_count=None
                )
                logger.info(f"Modified cell {mod.cell_index}: {mod.change_description}")

        return NotebookStructure(cells=cells)

    def _store_targeted_modification_in_mem0(
        self,
        request: ModifyRequest,
        target_index: int,
        all_indices: List[int],
        result: ModificationResult
    ) -> None:
        """Store targeted modification in Mem0."""
        try:
            # Store cell modification
            enhanced_memory.store_cell_modification(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                cell_name=request.cell_name,
                modification_details={
                    "cell_index": target_index,
                    "instruction": request.instruction,
                    "changes": result.changes_summary
                }
            )

            # Store dependency pattern if cascading changes occurred
            if len(all_indices) > 1:
                affected = [request.notebook.cells[i].cell_name or f"cell_{i}" for i in all_indices if i != target_index]
                enhanced_memory.store_dependency_pattern(
                    user_id=request.user_id,
                    notebook_id=request.notebook_id,
                    source_cell=request.cell_name,
                    affected_cells=affected,
                    reason=f"Modification: {request.instruction}"
                )

            # Store notebook context
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="targeted_modify",
                details={
                    "cell_name": request.cell_name,
                    "instruction": request.instruction,
                    "cells_modified": all_indices,
                    "changes": result.changes_summary
                }
            )

        except Exception as e:
            logger.error(f"Error storing in Mem0: {e}")

    def _store_notebook_modification_in_mem0(
        self,
        request: ModifyRequest,
        modified_indices: List[int],
        result: ModificationResult
    ) -> None:
        """Store notebook-level modification in Mem0."""
        try:
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="notebook_modify",
                details={
                    "instruction": request.instruction,
                    "cells_modified": modified_indices,
                    "changes": result.changes_summary
                }
            )
        except Exception as e:
            logger.error(f"Error storing in Mem0: {e}")
