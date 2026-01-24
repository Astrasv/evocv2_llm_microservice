"""Fix endpoint for repairing broken notebooks - stateless with Mem0."""

from fastapi import APIRouter, HTTPException, status
import logging

from app.models import FixRequest, FixResponse
from app.graph import workflow, WorkflowState


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["fix"])


@router.post("/fix", response_model=FixResponse)
async def fix_notebook(request: FixRequest):
    """
    Fix a broken notebook based on error traceback.

    Stateless operation - client sends broken notebook and receives fixed version.

    Features:
    - Intelligent error analysis using LLM
    - Mem0-enhanced with user's common error patterns
    - Retry loop with validation (up to 3 attempts)
    - Stores successful fix patterns for future learning

    Always uses full notebook context for comprehensive error fixing.
    """
    try:
        logger.info(f"Received fix request for notebook {request.notebook_id}")
        logger.info(f"User: {request.user_id}")

        # Execute workflow with retry loop
        state: WorkflowState = {
            "operation": "fix",
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

        notebook = final_state.get("notebook") or request.notebook
        fixes = final_state.get("changes_made", [])
        validation_passed = final_state.get("validation_passed", False)
        
        message = "Notebook fixed successfully" if validation_passed else "Fixes applied but validation incomplete"

        logger.info(f"Fix completed for notebook {request.notebook_id}: {message}")

        return FixResponse(
            notebook_id=request.notebook_id,
            notebook=notebook,
            fixes_applied=fixes,
            validation_passed=validation_passed,
            requirements=notebook.requirements,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fix endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
