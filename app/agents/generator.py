"""generator agent - creates 12-cell deap notebooks using llm with mem0 integration."""

import logging
from typing import Dict, Any

from app.models import GenerateRequest, NotebookStructure, NotebookCell
from app.core.request_parser import RequestParser
from app.utils.cell_names import CellNameMapper
from app.core.llm_cell_generator import LLMCellGenerator
from app.memory import enhanced_memory

logger = logging.getLogger(__name__)


class NotebookGenerator:
    """generates complete 12-cell deap notebooks using llm with mem0 storage."""

    def __init__(self):
        self.request_parser = RequestParser()
        self.cell_mapper = CellNameMapper()
        self.llm_generator = LLMCellGenerator()

    def generate(self, request: GenerateRequest) -> NotebookStructure:
        """
        generate a complete 12-cell notebook from specification.

        returns:
            - notebookstructure: the generated notebook
        """
        logger.info(f"generating notebook {request.notebook_id} for user {request.user_id}")

        # parse the flexible request format into structured data
        problem_data = self.request_parser.extract_structured_data(request)

        logger.info(f"parsed problem data: {problem_data.get('problem_name')}")

        # generate all cells in a single llm pass
        complete_notebook = self.llm_generator.generate_all_cells(problem_data)

        # convert llm response to notebookcell objects
        cells = []
        for cell_index, cell_data in enumerate(complete_notebook.cells):
            cell = NotebookCell(
                cell_type="code",
                cell_name=cell_data.cell_name,
                source=cell_data.source_code,
                execution_count=None,
                metadata={"cell_index": cell_index}
            )
            cells.append(cell)
            logger.info(f"processed cell {cell_index}: {cell_data.cell_name}")

        notebook = NotebookStructure(
            cells=cells,
            requirements=complete_notebook.requirements
        )

        # store generation in mem0
        self._store_generation_in_mem0(request, problem_data)

        logger.info(f"successfully generated notebook {request.notebook_id} for user {request.user_id}")

        return notebook

    def _store_generation_in_mem0(
        self,
        request: GenerateRequest,
        problem_data: Dict[str, Any]
    ) -> None:
        """store generation details in mem0."""
        try:
            # extract key information
            problem_name = problem_data.get('problem_name', 'Unknown')
            problem_type = self._infer_problem_type(problem_data)
            operators_used = self._extract_operators(problem_data)

            # store notebook context
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="generate",
                details={
                    "problem_name": problem_name,
                    "problem_type": problem_type,
                    "operators": operators_used,
                    "solution_size": problem_data.get('solution_size', 'unknown'),
                    "algorithm_type": problem_data.get('algorithm_type', 'simple')
                }
            )

            # store user preferences based on choices
            if operators_used.get('selection'):
                enhanced_memory.add_user_preference(
                    user_id=request.user_id,
                    preference_type="selection_operator",
                    value=operators_used['selection']
                )

            if operators_used.get('crossover'):
                enhanced_memory.add_user_preference(
                    user_id=request.user_id,
                    preference_type="crossover_operator",
                    value=operators_used['crossover']
                )

            if operators_used.get('mutation'):
                enhanced_memory.add_user_preference(
                    user_id=request.user_id,
                    preference_type="mutation_operator",
                    value=operators_used['mutation']
                )

            # store problem type preference
            if problem_type:
                enhanced_memory.add_user_preference(
                    user_id=request.user_id,
                    preference_type="problem_type",
                    value=problem_type
                )

        except Exception as e:
            logger.error(f"error storing generation in mem0: {e}")

    def _infer_problem_type(self, problem_data: Dict[str, Any]) -> str:
        """infer problem type from problem data."""
        problem_name = problem_data.get('problem_name', '').lower()
        solution_rep = problem_data.get('solution_representation', '').lower()

        if 'tsp' in problem_name or 'traveling' in problem_name:
            return 'combinatorial_tsp'
        elif 'knapsack' in problem_name:
            return 'combinatorial_knapsack'
        elif 'schedule' in problem_name or 'scheduling' in problem_name:
            return 'combinatorial_scheduling'
        elif 'permutation' in solution_rep:
            return 'combinatorial_permutation'
        elif 'binary' in solution_rep or 'bit' in solution_rep:
            return 'binary_optimization'
        elif 'real' in solution_rep or 'continuous' in solution_rep:
            return 'continuous_optimization'
        else:
            return 'general_optimization'

    def _extract_operators(self, problem_data: Dict[str, Any]) -> Dict[str, str]:
        """extract operators from problem data."""
        return {
            "selection": problem_data.get('selection_method', ''),
            "crossover": problem_data.get('crossover_operator', ''),
            "mutation": problem_data.get('mutation_operator', '')
        }
