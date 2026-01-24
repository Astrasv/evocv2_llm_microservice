"llm-based cell generation for deap notebooks."

from typing import Dict, Any, Optional
import instructor
from groq import Groq
from pydantic import BaseModel, Field
import logging

from app.config import settings
from app.utils.cell_names import CellNameMapper
from app.prompts.generator import (
    get_complete_notebook_prompt,
    get_single_cell_prompt,
    get_system_prompt_complete_generation,
    get_system_prompt_cell_generation
)

logger = logging.getLogger(__name__)


class CellGenerationResult(BaseModel):
    """structured result from llm cell generation."""
    source_code: str = Field(..., description="The Python source code for this cell")
    explanation: str = Field(..., description="Brief explanation of what this cell does")


class SingleCellCode(BaseModel):
    """a single cell's code in the complete notebook."""
    cell_name: str = Field(..., description="Name of the cell (e.g., 'imports', 'config', 'creator', etc.)")
    source_code: str = Field(..., description="The Python source code for this cell")


class CompleteNotebookGeneration(BaseModel):
    """complete notebook with all 12 cells generated in a single pass."""
    cells: list[SingleCellCode] = Field(..., min_length=12, max_length=12, description="All 12 cells in order")
    requirements: str = Field(..., description="Newline-separated packages needed for the code (e.g., 'numpy\ndeap\nmatplotlib')")


class LLMCellGenerator:
    """generates individual notebook cells using llm."""

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON,
            model=settings.model_name
        )
        self.cell_mapper = CellNameMapper()

    def generate_all_cells(self, problem_data: Dict[str, Any]) -> CompleteNotebookGeneration:
        """
        generate all 12 cells in a single llm pass.

        args:
            problem_data: structured problem data from requestparser

        returns:
            completenotebookgeneration with all 12 cells
        """
        logger.info("generating all 12 cells in a single llm pass")

        # build comprehensive prompt for all cells
        prompt = get_complete_notebook_prompt(problem_data)

        try:
            result: CompleteNotebookGeneration = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": get_system_prompt_complete_generation()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_model=CompleteNotebookGeneration,
                temperature=0.3,
                max_tokens=8000
            )

            logger.info(f"successfully generated all 12 cells in single pass")
            return result

        except Exception as e:
            logger.error(f"llm complete notebook generation failed: {e}")
            # fallback to basic templates
            return self._fallback_complete_notebook(problem_data)

    def generate_cell(
        self,
        cell_index: int,
        problem_data: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        generate a single cell using llm.

        args:
            cell_index: index of the cell to generate (0-11)
            problem_data: structured problem data from requestparser
            context: optional context from previously generated cells

        returns:
            source code for the cell
        """
        cell_name = self.cell_mapper.get_cell_name(cell_index)
        cell_description = self.cell_mapper.get_cell_description(cell_name)

        logger.info(f"generating cell {cell_index} ({cell_name}) using llm")

        # build context-aware prompt
        prompt = get_single_cell_prompt(cell_index, cell_name, cell_description, problem_data, context)

        try:
            result: CellGenerationResult = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": get_system_prompt_cell_generation()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_model=CellGenerationResult,
                temperature=0.3,
                max_tokens=2000
            )

            logger.info(f"successfully generated cell {cell_index}: {result.explanation}")
            return result.source_code

        except Exception as e:
            logger.error(f"llm cell generation failed for cell {cell_index}: {e}")
            # fallback to basic template
            return self._fallback_template(cell_index, problem_data)

    def _fallback_complete_notebook(self, problem_data: Dict[str, Any]) -> CompleteNotebookGeneration:
        """generate fallback complete notebook using templates."""

        cell_names = [
            "imports", "config", "creator", "evaluate", "crossover",
            "mutation", "selection", "additional_operators", "initialization",
            "toolbox_registration", "evolution_loop", "results_and_plots"
        ]

        cells = []
        for i, name in enumerate(cell_names):
            source = self._fallback_template(i, problem_data)
            cells.append(SingleCellCode(cell_name=name, source_code=source))

        return CompleteNotebookGeneration(cells=cells, requirements="deap\nnumpy\nmatplotlib")

    def _fallback_template(self, cell_index: int, problem_data: Dict[str, Any]) -> str:
        """fallback templates when llm fails."""

        templates = {
            0: """from deap import base, creator, tools, algorithms
import numpy as np
import random""",
            1: f"""# problem configuration
DIMENSIONS = {problem_data.get('solution_size', 10)}
LOWER_BOUND = {problem_data.get('lower_bounds', [0])}
UPPER_BOUND = {problem_data.get('upper_bounds', [1])}

random.seed(42)
np.random.seed(42)""",
            2: """# create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)""",
            3: """def evaluate(individual):
    return sum(x**2 for x in individual),""",
            4: """def mate(ind1, ind2):
    tools.cxBlend(ind1, ind2, 0.5)
    return ind1, ind2""",
            5: """def mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
    return individual,""",
            6: """def select(individuals, k):
    return tools.selTournament(individuals, k, tournsize=3)""",
            7: """# additional custom operators can be defined here""",
            8: """def create_individual():
    return creator.Individual([
        random.uniform(LOWER_BOUND[i], UPPER_BOUND[i])
        for i in range(DIMENSIONS)
    ])""",
            9: """toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)""",
            10: f"""population = toolbox.population(n={problem_data.get('population_size', 100)})

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(10)

population, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb={problem_data.get('crossover_probability', 0.7)},
    mutpb={problem_data.get('mutation_probability', 0.2)},
    ngen={problem_data.get('num_generations', 50)},
    stats=stats,
    halloffame=hof,
    verbose=True
)""",
            11: """print("\nBest individuals:")
for i, ind in enumerate(hof, 1):
    print(f"{i}. Fitness: {ind.fitness.values[0]:.6f}")

print("\nFinal Statistics:")
record = logbook[-1]
print(f"Min: {record['min']:.6f}")
print(f"Avg: {record['avg']:.6f}")"""
        }

        return templates.get(cell_index, f"# Cell {cell_index}")
