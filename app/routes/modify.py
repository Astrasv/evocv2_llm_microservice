"""Modify endpoint for updating existing notebooks - stateless with Mem0."""

from fastapi import APIRouter, HTTPException, status
import logging

from app.models import ModifyRequest, ModifyResponse
from app.graph import workflow, WorkflowState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["modify"])


@router.post("/modify", response_model=ModifyResponse)
async def modify_notebook(request: ModifyRequest):
    """
    Modify an existing notebook using natural language instructions.

    Stateless operation - client sends full notebook and receives modified version.

    Supports two modification modes:
    1. Targeted cell modification: Specify cell_name (e.g., 'mutate', 'crossover')
       - Uses minimal context (70-85% token savings)
       - Dynamic dependency detection via LLM
       - Mem0 provides user preferences and learned patterns

    2. Notebook-level modification: No cell_name specified
       - Uses full notebook context
       - For complex/generic changes
       - Mem0 enhances with user patterns
    """
    try:
        logger.info(f"Received modify request for notebook {request.notebook_id}")
        logger.info(f"User: {request.user_id}, Cell: {request.cell_name or 'notebook-level'}")

        # Execute workflow
        state: WorkflowState = {
            "operation": "modify",
            "user_id": request.user_id,
            "notebook_id": request.notebook_id,
            "request": request,
            "notebook": None,
            "changes_made": [],
            "cells_modified": [],
            "validation_passed": False,
            "error": None,
            "retry_count": 0,
            "max_retries": 3
        }

        final_state = workflow.execute(state)

        if final_state.get("error") and not final_state.get("notebook"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Modification failed: {final_state['error']}"
            )

        notebook = final_state["notebook"]
        changes = final_state.get("changes_made", [])
        cells_modified = final_state.get("cells_modified", [])

        logger.info(f"Successfully modified notebook {request.notebook_id}")
        logger.info(f"Modified cells: {cells_modified}")

        return ModifyResponse(
            notebook_id=request.notebook_id,
            notebook=notebook,
            changes_made=changes,
            cells_modified=cells_modified,
            message="Notebook modified successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Modify endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
