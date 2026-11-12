"""Small, dependency-free run-state helpers shared by Agent and Evaluator."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os


AGENT_RUN_STATE_FILENAME = "agent_run_state.json"


def write_agent_run_state(
    run_results_dir: str,
    status: str,
    *,
    run_name: str,
    model: str,
    selected_tasks: int,
    pending_tasks: int,
) -> None:
    """Atomically publish whether the complete Agent selection has finished."""
    payload = {
        "status": status,
        "run_name": run_name,
        "model": model,
        "selected_tasks": selected_tasks,
        "pending_tasks_at_start": pending_tasks,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(run_results_dir, AGENT_RUN_STATE_FILENAME)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    os.replace(temp_path, path)


def agent_run_is_complete(agent_outputs: str) -> bool:
    """Return whether Agent has published a complete run for this directory."""
    state_path = os.path.join(agent_outputs, AGENT_RUN_STATE_FILENAME)
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        state = None
    except (OSError, ValueError, TypeError):
        return False

    if isinstance(state, dict):
        return state.get("status") == "complete"

    # Compatibility for runs produced before agent_run_state.json existed.
    # A resumed local run rewrites runtime.json before regenerating its final
    # summary, so a newer runtime file means the old summary is stale.
    summary_path = os.path.join(agent_outputs, "agent_summary.json")
    if not os.path.isfile(summary_path):
        return False
    runtime_path = os.path.join(agent_outputs, "runtime.json")
    if os.path.isfile(runtime_path):
        try:
            if os.path.getmtime(runtime_path) > os.path.getmtime(summary_path):
                return False
        except OSError:
            return False
    return True
