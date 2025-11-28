"""LangGraph workflow for notebook operations - stateless with Mem0."""

from typing import TypedDict, Literal, Optional, List, Any
from langgraph.graph import StateGraph, END
from app.models import GenerateRequest, ModifyRequest, FixRequest, NotebookStructure
from app.agents.generator import NotebookGenerator
from app.agents.modifier import NotebookModifier
from app.agents.fixer import NotebookFixer
import logging

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """Shared state for LangGraph workflow - stateless."""
    operation: Literal["generate", "modify", "fix"]
    user_id: str
    notebook_id: str
    request: Optional[Any]
    notebook: Optional[NotebookStructure]
    changes_made: List[str]
    cells_modified: List[int]
    validation_passed: bool
    error: Optional[str]
    retry_count: int
    max_retries: int


class NotebookWorkflow:
    """LangGraph workflow for stateless notebook operations with Mem0."""

    def __init__(self):
        self.generator = NotebookGenerator()
        self.modifier = NotebookModifier()
        self.fixer = NotebookFixer()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        # Add operation nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("modify", self._modify_node)
        workflow.add_node("fix", self._fix_node)

        # Set conditional entry point
        workflow.set_conditional_entry_point(
            self._route_entry,
            {
                "generate": "generate",
                "modify": "modify",
                "fix": "fix"
            }
        )

        # All operations go directly to END (stateless, no validation retries)
        workflow.add_edge("generate", END)
        workflow.add_edge("modify", END)
        workflow.add_edge("fix", END)

        return workflow.compile()

    def _route_entry(self, state: WorkflowState) -> str:
        """Route to appropriate operation."""
        return state["operation"]

    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        """Generate notebook node."""
        try:
            request: GenerateRequest = state["request"]
            logger.info(f"Executing generate for user {state['user_id']}")

            # Generate notebook (includes Mem0 storage)
            notebook = self.generator.generate(request)

            state["notebook"] = notebook
            state["validation_passed"] = True

            logger.info(f"Generate completed: {state['notebook_id']}")

        except Exception as e:
            logger.error(f"Generate failed: {e}", exc_info=True)
            state["error"] = str(e)
            state["validation_passed"] = False

        return state

    def _modify_node(self, state: WorkflowState) -> WorkflowState:
        """Modify notebook node."""
        try:
            request: ModifyRequest = state["request"]
            logger.info(f"Executing modify for notebook {state['notebook_id']}")

            # Modify notebook (includes Mem0 storage)
            notebook, changes, cells_modified = self.modifier.modify(request)

            state["notebook"] = notebook
            state["changes_made"] = changes
            state["cells_modified"] = cells_modified
            state["validation_passed"] = True

            logger.info(f"Modify completed: {len(changes)} changes")

        except Exception as e:
            logger.error(f"Modify failed: {e}", exc_info=True)
            state["error"] = str(e)
            state["validation_passed"] = False

        return state

    def _fix_node(self, state: WorkflowState) -> WorkflowState:
        """Fix notebook node."""
        try:
            request: FixRequest = state["request"]
            logger.info(f"Executing fix for notebook {state['notebook_id']}")

            # Fix notebook with retries (includes Mem0 storage)
            notebook, fixes, validation_passed = self.fixer.fix(
                request,
                max_retries=state.get("max_retries", 3)
            )

            state["notebook"] = notebook
            state["changes_made"] = fixes
            state["validation_passed"] = validation_passed

            if validation_passed:
                logger.info(f"Fix completed successfully")
            else:
                logger.warning(f"Fix completed with validation issues")

        except Exception as e:
            logger.error(f"Fix failed: {e}", exc_info=True)
            state["error"] = str(e)
            state["validation_passed"] = False

        return state

    def execute(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute the workflow."""
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            initial_state["error"] = str(e)
            return initial_state


# Global workflow instance
workflow = NotebookWorkflow()
