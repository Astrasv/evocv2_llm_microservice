from typing import Dict, List

class CellNameMapper:
    """maps cell indices to descriptive names and provides metadata."""

    # standard 12-cell structure
    CELL_NAMES = {
        0: "imports",
        1: "config",
        2: "creator",
        3: "evaluate",
        4: "crossover",
        5: "mutation",
        6: "selection",
        7: "additional_operators",
        8: "initialization",
        9: "toolbox_registration",
        10: "evolution_loop",
        11: "results_and_plots"
    }

    CELL_DESCRIPTIONS = {
        "imports": "import all required libraries (deap, numpy, matplotlib, etc.)",
        "config": "problem configuration (dimensions, bounds, random seed)",
        "creator": "create fitness and individual classes using creator.create",
        "evaluate": "define the objective/fitness evaluation function",
        "crossover": "define the crossover/mating function",
        "mutation": "define the mutation function",
        "selection": "define the selection function",
        "additional_operators": "define any additional custom operators",
        "initialization": "define individual initialization function",
        "toolbox_registration": "register all operators in the deap toolbox",
        "evolution_loop": "main evolutionary algorithm loop",
        "results_and_plots": "display results, best solutions, and generate plots"
    }

    @classmethod
    def get_cell_name(cls, index: int) -> str:
        return cls.CELL_NAMES.get(index, f"cell_{index}")

    @classmethod
    def get_cell_description(cls, cell_name: str) -> str:
        return cls.CELL_DESCRIPTIONS.get(cell_name, "")

    @classmethod
    def get_index_by_name(cls, cell_name: str) -> int:
        for idx, name in cls.CELL_NAMES.items():
            if name == cell_name:
                return idx
        return -1

    @classmethod
    def get_all_cell_names(cls) -> List[str]:
        return [cls.CELL_NAMES[i] for i in range(12)]

    @classmethod
    def get_cell_metadata(cls, index: int) -> Dict[str, str]:
        name = cls.get_cell_name(index)
        return {
            "index": index,
            "name": name,
            "description": cls.CELL_DESCRIPTIONS.get(name, "")
        }


# dependency mapping: which cells depend on which other cells
CELL_DEPENDENCIES = {
    "toolbox_registration": ["evaluate", "crossover", "mutation", "selection", "initialization"],
    "evolution_loop": ["toolbox_registration", "config"],
    "results_and_plots": ["evolution_loop"]
}


def get_dependent_cells(cell_name: str) -> List[str]:
    return CELL_DEPENDENCIES.get(cell_name, [])


def get_cells_dependent_on(cell_name: str) -> List[str]:
    dependents = []
    for cell, deps in CELL_DEPENDENCIES.items():
        if cell_name in deps:
            dependents.append(cell)
    return dependents
