"""Run configuration for the SMDD-Bench agent harness.

Edit the defaults below to configure the agent harness. Frequently adjusted
LLM limits can also be overridden by the runner CLI.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    model: str = "azure/gpt-5.4"
    max_turns: int = 100
    max_output_tokens: Optional[int] = None  # per LLM response
    max_consecutive_no_tool: Optional[int] = None
    llm_timeout: Optional[int] = None  # seconds per LLM request
    max_tokens: int = 10_000_000  # cumulative input + output
    boltz_budget: int = 8
    boltz_max_repeats_per_request: int = 2
    admet_budget: int = 15
    vina_budget: int = 20
    diffdock_budget: int = 8
    temperature: float = 1.0
    python_timeout: int = 300
    boltz_timeout: int = 1800  # 30 min
    boltz_worker_startup_timeout: int = 3600
    boltz_cpu_threads: int = 16
    admet_timeout: int = 60
    vina_timeout: int = 900
    diffdock_timeout: int = 3600
    task_timeout: int = 43_200  # hard per-task ceiling for local supervision
    max_refusals: int = 15
    seed: int = 42
    docker_image: str = "smdd-agent"
    use_responses_api: bool = False
