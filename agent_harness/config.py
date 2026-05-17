"""Run configuration for the SMDD-Bench agent harness.

Edit the defaults below to configure the agent harness.
These values are the single source of truth — no CLI overrides.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    model: str = "azure/gpt-5.4"
    max_turns: int = 100
    max_tokens: int = 10_000_000  # cumulative input + output
    boltz_budget: int = 8
    admet_budget: int = 15
    temperature: float = 1.0
    python_timeout: int = 300
    boltz_timeout: int = 1800  # 30 min
    admet_timeout: int = 60
    max_refusals: int = 15
    seed: int = 42
    docker_image: str = "smdd-agent"
    use_responses_api: bool = False
