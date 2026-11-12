"""Fail-fast checks shared by the local Agent and evaluator runners."""

from __future__ import annotations

import os


def validate_gpu_ids(gpu_ids: list[int], available_gpu_ids: list[int]) -> list[int]:
    """Validate explicit CUDA device IDs against the host-visible devices."""
    if not gpu_ids:
        raise RuntimeError("No GPUs detected or selected; Boltz requires a CUDA GPU")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise RuntimeError(f"GPU IDs must be non-negative integers: {gpu_ids}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"GPU IDs must not be repeated: {gpu_ids}")

    available = set(available_gpu_ids)
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in available]
    if missing:
        raise RuntimeError(
            f"Selected GPU IDs are not visible: {missing}; "
            f"nvidia-smi reports {available_gpu_ids}"
        )
    return gpu_ids


def validate_agent_config(config) -> None:
    """Reject values that would otherwise fail late or wait indefinitely."""
    if not str(config.model).strip():
        raise RuntimeError("Agent model name is empty")
    for name in (
        "max_turns",
        "max_tokens",
        "boltz_timeout",
        "boltz_worker_startup_timeout",
        "boltz_cpu_threads",
        "task_timeout",
    ):
        if int(getattr(config, name)) < 1:
            raise RuntimeError(f"Agent setting {name} must be positive")
    for name in ("llm_timeout", "max_output_tokens", "max_consecutive_no_tool"):
        value = getattr(config, name)
        if value is not None and int(value) < 1:
            raise RuntimeError(f"Agent setting {name} must be positive when configured")
    if config.llm_timeout is not None and config.llm_timeout > config.task_timeout:
        raise RuntimeError(
            f"LLM timeout ({config.llm_timeout}s) exceeds the per-task timeout "
            f"({config.task_timeout}s)"
        )
    if config.boltz_timeout > config.task_timeout:
        raise RuntimeError(
            f"Boltz timeout ({config.boltz_timeout}s) exceeds the per-task timeout "
            f"({config.task_timeout}s)"
        )


def _redact_secrets(message: str) -> str:
    redacted = str(message)
    for name, value in os.environ.items():
        upper = name.upper()
        if value and any(marker in upper for marker in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            redacted = redacted.replace(value, "***")
    return redacted


def check_model_api(model: str, use_responses_api: bool, timeout: int | None) -> None:
    """Make one minimal, retry-free model request before scheduling tasks."""
    import litellm

    request_timeout = min(max(5, int(timeout or 60)), 120)
    try:
        if use_responses_api:
            response = litellm.responses(
                model=model,
                input=[{"role": "user", "content": "Reply OK."}],
                max_output_tokens=1,
                temperature=0,
                timeout=request_timeout,
                num_retries=0,
            )
        else:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Reply OK."}],
                max_tokens=1,
                temperature=0,
                timeout=request_timeout,
                num_retries=0,
            )
        if response is None:
            raise RuntimeError("model endpoint returned no response")
    except Exception as exc:
        detail = _redact_secrets(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Model API preflight failed for {model!r} within "
            f"{request_timeout}s: {detail}"
        ) from exc
