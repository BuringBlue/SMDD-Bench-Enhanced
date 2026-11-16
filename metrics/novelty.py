"""Novelty metrics for lead-optimization candidates."""

from __future__ import annotations

from metrics.lead_optimization import read_smiles_file
from smdd_runtime.novelty import DEFAULT_SEARCH_FLOOR, request_novelty


def assess_novelty_task4(
    agent_file_path: str,
    max_database_similarity: float | None = None,
    search_floor: float = DEFAULT_SEARCH_FLOOR,
    timeout: int = 300,
) -> dict:
    """Record database and scaffold novelty, with an optional similarity gate.

    Scaffold novelty is deliberately diagnostic and never changes
    ``all_passed``. Database similarity is also record-only unless callers
    explicitly provide ``max_database_similarity``.
    """
    if max_database_similarity is not None:
        if not search_floor <= max_database_similarity <= 1.0:
            raise ValueError(
                "max_database_similarity must be in [search_floor, 1]"
            )

    result = request_novelty(
        read_smiles_file(agent_file_path),
        search_floor=search_floor,
        timeout=timeout,
    )
    similarities = {
        name: source["max_similarity"]
        for name, source in result["database_similarity"].items()
    }
    observed = [value for value in similarities.values() if value is not None]
    passed = (
        max_database_similarity is None
        or not observed
        or max(observed) < max_database_similarity
    )
    return {
        "all_passed": passed,
        "mode": "record_only" if max_database_similarity is None else "threshold",
        "max_database_similarity": max_database_similarity,
        "observed_max_similarity": max(observed) if observed else None,
        **result,
    }
