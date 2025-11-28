"""Generate endpoint for creating new DEAP notebooks - stateless with Mem0."""

from fastapi import APIRouter, HTTPException, status
import logging

from app.models import GenerateRequest, GenerateResponse
from app.graph import workflow, WorkflowState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["generate"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest):
    """
    Generate a new 12-cell DEAP notebook from specification.

    Stateless operation - client receives notebook and must manage it.
    User preferences and patterns are stored in Mem0 for personalization.
    """
    try:
        logger.info(f"Received generate request for user {request.user_id}")

        # Execute workflow
        state: WorkflowState = {
            "operation": "generate",
            "user_id": request.user_id,
            "notebook_id": request.notebook_id,
            "request": request,
            "notebook": None,
            "changes_made": [],
            "validation_passed": False,
            "error": None,
            "retry_count": 0,
            "max_retries": 3
        }

        final_state = workflow.execute(state)

        if final_state.get("error") and not final_state.get("notebook"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {final_state['error']}"
            )

        notebook = final_state["notebook"]

        logger.info(f"Successfully generated notebook {request.notebook_id} for user {request.user_id}")

        return GenerateResponse(
            notebook_id=request.notebook_id,
            notebook=notebook,
            message="Notebook generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
