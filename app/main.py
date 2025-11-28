"""FastAPI application for EVOC DEAP Agent - Stateless with Mem0."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.routes import generate_router, modify_router, fix_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("Starting EVOC DEAP Agent (Stateless + Mem0)")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Mem0 API Key: {'Configured' if settings.mem0_api_key else 'Not configured (using local ChromaDB)'}")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down EVOC DEAP Agent service")


app = FastAPI(
    title="EVOC DEAP Agent",
    description="""
    Stateless LLM service for generating and maintaining DEAP evolutionary algorithms with Mem0 personalization.

    ## Architecture
    * **Stateless** - No server-side sessions. Client manages notebook state.
    * **Mem0 Integration** - User preferences and patterns stored in Mem0
    * **Dual-Level Context** - User-level and notebook-level memory
    * **3-Case Modification** - Targeted, notebook-level, and error fixing

    ## Features
    * **Generate** - Create complete 12-cell DEAP notebooks from flexible specifications
    * **Modify** - Update notebooks with natural language (targeted or notebook-level)
    * **Fix** - Automatically repair broken notebooks from error tracebacks

    ## Key Improvements
    * **Token Efficiency** - Targeted modifications save 70-85% on tokens
    * **Smart Dependencies** - Dynamic LLM-based dependency detection
    * **Learning System** - Mem0 stores patterns, preferences, and common errors
    * **No Sessions** - True horizontal scalability

    ## API Design
    All requests require:
    - `user_id` - Mem0's user_id for personalization (user-level context)
    - `notebook_id` - Mem0's run_id for tracking (notebook-level context)
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    tags_metadata=[
        {
            "name": "generate",
            "description": "Generate new DEAP notebooks from specifications"
        },
        {
            "name": "modify",
            "description": "Modify existing notebooks (targeted or notebook-level)"
        },
        {
            "name": "fix",
            "description": "Fix broken notebooks from error tracebacks"
        }
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "EVOC DEAP Agent",
        "version": "3.0.0",
        "architecture": "stateless",
        "status": "running",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "generate": "POST /v1/generate",
            "modify": "POST /v1/modify",
            "fix": "POST /v1/fix"
        },
        "features": {
            "stateless": True,
            "mem0_integration": True,
            "user_level_context": True,
            "notebook_level_context": True,
            "targeted_modifications": True,
            "dynamic_dependencies": True,
            "token_efficient": True,
            "single_pass_generation": True,
            "flexible_input": True,
            "error_pattern_learning": True
        },
        "memory": {
            "user_preferences": "Mem0 user_id",
            "notebook_tracking": "Mem0 run_id",
            "cell_patterns": "Learned over time",
            "dependencies": "Dynamically detected",
            "error_patterns": "Stored and reused"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "architecture": "stateless + mem0"
    }


# Include routers
app.include_router(generate_router)
app.include_router(modify_router)
app.include_router(fix_router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False
    )
