"""Memory layer using Mem0 cloud with ChromaDB fallback."""

from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class EnhancedMemory:
    """
    Manages user and notebook-level memory using Mem0.

    - Uses Mem0 cloud (MemoryClient) if API key is available
    - Falls back to local ChromaDB if no API key
    - user_id → Mem0's user_id
    - notebook_id → Mem0's run_id
    """

    def __init__(self):
        """Initialize Mem0 memory with cloud or local fallback."""
        from app.config import settings

        self.memory = None
        self._fallback_memory: Dict[str, List[Dict]] = {}

        # Try cloud first (MemoryClient)
        if settings.mem0_api_key:
            try:
                from mem0 import MemoryClient
                self.memory = MemoryClient(
                    api_key=settings.mem0_api_key,
                    org_id=settings.mem0_org_id,
                    project_id=settings.mem0_project_id
                )
                
                logger.info("✓ Mem0 cloud initialized successfully (MemoryClient)")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Mem0 cloud: {e}")
                logger.info("Falling back to local ChromaDB...")

        # Fallback to local ChromaDB
        try:
            from mem0 import Memory
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "evoc_deap_memory",
                        "path": "/tmp/chroma_db"
                    }
                }
            }
            self.memory = Memory.from_config(config)
            logger.info("✓ Mem0 local ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 (cloud and local): {e}")
            logger.warning("Using in-memory fallback (no persistence)")
            self.memory = None

    # ==================== User-Level Methods ====================

    def add_user_preference(
        self,
        user_id: str,
        preference_type: str,
        value: Any,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store user-level preference."""
        message = f"User preference - {preference_type}: {value}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    metadata={
                        "type": "user_preference",
                        "preference_type": preference_type,
                        **(metadata or {})
                    }
                )
                logger.debug(f"Stored user preference for {user_id}: {preference_type}")
            else:
                self._fallback_add(user_id, "user_preference", message)
        except Exception as e:
            logger.error(f"Failed to add user preference: {e}")

    def get_user_preferences(
        self,
        user_id: str,
        preference_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve user-level preferences, optionally filtered by type."""
        try:
            if self.memory:
                response = self.memory.get_all(filters={"user_id": user_id})
                memories = response.get("results", []) if isinstance(response, dict) else response
                results = [
                    m for m in memories
                    if m.get("metadata", {}).get("type") == "user_preference"
                ]

                if preference_type:
                    results = [
                        r for r in results
                        if r.get("metadata", {}).get("preference_type") == preference_type
                    ]

                return results
            else:
                return self._fallback_get(user_id, "user_preference")
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return []

    def search_user_context(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search user-level context using semantic similarity."""
        try:
            if self.memory:
                response = self.memory.search(
                    query,
                    filters={"user_id": user_id},
                    limit=limit
                )
                return response.get("results", []) if isinstance(response, dict) else response
            else:
                return self._fallback_get(user_id, None)[:limit]
        except Exception as e:
            logger.error(f"Failed to search user context: {e}")
            return []

    # ==================== Notebook-Level Methods ====================

    def add_notebook_context(
        self,
        user_id: str,
        notebook_id: str,
        operation: str,
        details: Dict[str, Any]
    ) -> None:
        """Store notebook-level context using run_id."""
        message = f"Notebook operation - {operation}: {json.dumps(details, indent=2)}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    run_id=notebook_id,
                    metadata={
                        "type": "notebook_context",
                        "operation": operation,
                        "notebook_id": notebook_id
                    }
                )
                logger.debug(f"Stored notebook context: {notebook_id} - {operation}")
            else:
                self._fallback_add(f"{user_id}:{notebook_id}", "notebook_context", message)
        except Exception as e:
            logger.error(f"Failed to add notebook context: {e}")

    def get_notebook_history(
        self,
        user_id: str,
        notebook_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get chronological history for a specific notebook."""
        try:
            if self.memory:
                response = self.memory.get_all(
                    filters={"user_id": user_id, "run_id": notebook_id}
                )
                memories = response.get("results", []) if isinstance(response, dict) else response
                return memories[-limit:] if memories else []
            else:
                return self._fallback_get(f"{user_id}:{notebook_id}", "notebook_context")[-limit:]
        except Exception as e:
            logger.error(f"Failed to get notebook history: {e}")
            return []

    def search_notebook_context(
        self,
        user_id: str,
        notebook_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific notebook's context."""
        try:
            if self.memory:
                response = self.memory.search(
                    query,
                    filters={"user_id": user_id, "run_id": notebook_id},
                    limit=limit
                )
                return response.get("results", []) if isinstance(response, dict) else response
            else:
                return self._fallback_get(f"{user_id}:{notebook_id}", "notebook_context")[:limit]
        except Exception as e:
            logger.error(f"Failed to search notebook context: {e}")
            return []

    # ==================== Cell-Specific Methods ====================

    def store_cell_modification(
        self,
        user_id: str,
        notebook_id: str,
        cell_name: str,
        modification_details: Dict[str, Any]
    ) -> None:
        """Store cell-specific modification patterns."""
        message = f"Cell modification - {cell_name}: {json.dumps(modification_details, indent=2)}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    run_id=notebook_id,
                    metadata={
                        "type": "cell_modification",
                        "cell_name": cell_name,
                        "notebook_id": notebook_id,
                        **modification_details
                    }
                )
                logger.debug(f"Stored cell modification: {cell_name} in {notebook_id}")
            else:
                self._fallback_add(
                    f"{user_id}:{notebook_id}",
                    "cell_modification",
                    message
                )
        except Exception as e:
            logger.error(f"Failed to store cell modification: {e}")

    def get_cell_patterns(
        self,
        user_id: str,
        cell_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get user's patterns for a specific cell type across all notebooks."""
        try:
            if self.memory:
                response = self.memory.search(
                    f"cell modifications for {cell_name}",
                    filters={"user_id": user_id},
                    limit=limit
                )
                results = response.get("results", []) if isinstance(response, dict) else response
                return [
                    r for r in results
                    if r.get("metadata", {}).get("cell_name") == cell_name
                ]
            else:
                all_data = self._fallback_get(user_id, "cell_modification")
                return [d for d in all_data if cell_name in d.get("content", "")][:limit]
        except Exception as e:
            logger.error(f"Failed to get cell patterns: {e}")
            return []

    # ==================== Dependency Learning ====================

    def store_dependency_pattern(
        self,
        user_id: str,
        notebook_id: str,
        source_cell: str,
        affected_cells: List[str],
        reason: str
    ) -> None:
        """Learn and store dependency patterns."""
        details = {
            "source_cell": source_cell,
            "affected_cells": affected_cells,
            "reason": reason
        }
        message = f"Dependency pattern: Modifying {source_cell} affected {affected_cells} because {reason}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    run_id=notebook_id,
                    metadata={
                        "type": "dependency_pattern",
                        "notebook_id": notebook_id,
                        **details
                    }
                )
                logger.debug(f"Stored dependency pattern: {source_cell} -> {affected_cells}")
            else:
                self._fallback_add(f"{user_id}:{notebook_id}", "dependency_pattern", message)
        except Exception as e:
            logger.error(f"Failed to store dependency pattern: {e}")

    def get_learned_dependencies(
        self,
        user_id: str,
        cell_name: str
    ) -> List[str]:
        """Get learned dependencies for a cell based on past patterns."""
        try:
            if self.memory:
                response = self.memory.search(
                    f"dependencies when modifying {cell_name}",
                    filters={"user_id": user_id},
                    limit=5
                )
                results = response.get("results", []) if isinstance(response, dict) else response

                # Extract affected cells from metadata
                affected = set()
                for r in results:
                    metadata = r.get("metadata", {})
                    if metadata.get("source_cell") == cell_name:
                        affected.update(metadata.get("affected_cells", []))

                return list(affected)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get learned dependencies: {e}")
            return []

    # ==================== Error Pattern Tracking ====================

    def store_error_pattern(
        self,
        user_id: str,
        notebook_id: str,
        error_type: str,
        error_location: str,
        fix_applied: str
    ) -> None:
        """Store common error patterns and their fixes."""
        details = {
            "error_type": error_type,
            "error_location": error_location,
            "fix_applied": fix_applied
        }
        message = f"Error pattern: {error_type} at {error_location}, fixed by {fix_applied}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    run_id=notebook_id,
                    metadata={
                        "type": "error_pattern",
                        "notebook_id": notebook_id,
                        **details
                    }
                )
                logger.debug(f"Stored error pattern: {error_type}")
            else:
                self._fallback_add(f"{user_id}:{notebook_id}", "error_pattern", message)
        except Exception as e:
            logger.error(f"Failed to store error pattern: {e}")

    def get_common_errors(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get user's common error patterns."""
        try:
            if self.memory:
                response = self.memory.search(
                    "common errors and fixes",
                    filters={"user_id": user_id},
                    limit=limit
                )
                results = response.get("results", []) if isinstance(response, dict) else response
                return [
                    r for r in results
                    if r.get("metadata", {}).get("type") == "error_pattern"
                ]
            else:
                return self._fallback_get(user_id, "error_pattern")[:limit]
        except Exception as e:
            logger.error(f"Failed to get common errors: {e}")
            return []

    # ==================== Fallback Memory ====================

    def _fallback_add(self, key: str, data_type: str, content: str) -> None:
        """Fallback in-memory storage."""
        if key not in self._fallback_memory:
            self._fallback_memory[key] = []
        self._fallback_memory[key].append({
            "type": data_type,
            "content": content
        })

    def _fallback_get(self, key: str, data_type: Optional[str]) -> List[Dict]:
        """Fallback in-memory retrieval."""
        data = self._fallback_memory.get(key, [])
        if data_type:
            return [d for d in data if d.get("type") == data_type]
        return data


# Global memory instance
enhanced_memory = EnhancedMemory()
