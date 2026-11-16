"""ReAct loop for the SMDD-Bench agent."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from agent_harness.config import AgentConfig
from agent_harness.logger import AgentLogger
from agent_harness.prompts import TOOL_SCHEMAS, build_system_prompt

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_DISABLE_TELEMETRY", "True")
import litellm
from agent_harness.tools import ToolExecutor

litellm.drop_params = True


@dataclass
class _SubmitRetryGuard:
    path: Optional[str] = None
    failures: int = 0

    @staticmethod
    def key(arguments) -> str:
        path = str(arguments.get("file_path", "")).strip()
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        return os.path.normpath(path) if path else ""

    def reset(self) -> None:
        self.path = None
        self.failures = 0

    def blocked(self, path: str) -> bool:
        return path == self.path and self.failures >= 2

    def record_failure(self, path: str) -> int:
        if path == self.path:
            self.failures += 1
        else:
            self.path = path
            self.failures = 1
        return self.failures


SUBMIT_CORRECTION = (
    "Do not call submit() for this path again until the file exists. "
    "Use the python tool to write the requested output file in the workspace, "
    "then call submit()."
)

# ---------------------------------------------------------------------------
# Responses API caller (default)
# ---------------------------------------------------------------------------

def call_llm(model, input_items, tools, instructions, temperature=0.0,
             max_output_tokens=None, timeout=None):
    """Call an LLM via LiteLLM Responses API. Returns the response object.

    Args:
        model: LiteLLM model string (e.g. "gpt-4o", "claude-sonnet-4-20250514", "ollama/llama3.1")
        input_items: List of input item dicts (Responses API format)
        tools: List of tool schema dicts (Responses API format)
        instructions: System prompt string
        temperature: Sampling temperature
    """
    kwargs = dict(
        model=model,
        input=input_items,
        tools=tools,
        instructions=instructions,
        temperature=temperature,
        reasoning = {
            "effort": "high"
        },
        text={"verbosity": "high"},
        # thinking={"type":"enabled"},
        num_retries=25,
    )
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    return litellm.responses(**kwargs)


# ---------------------------------------------------------------------------
# Completions API caller + adapters
# ---------------------------------------------------------------------------

def _responses_input_to_messages(input_items, system_prompt):
    """Convert responses-format input_items to completions-format messages.

    Also injects Anthropic cache_control breakpoints on the system message
    and the last user/tool message.
    """
    messages = [
        {"role": "system", "content": system_prompt, "cache_control": {"type": "ephemeral"}},
    ]

    # We need to merge consecutive function_call items into a single assistant
    # message's tool_calls list, and handle the interleaved structure.
    i = 0
    while i < len(input_items):
        item = input_items[i]

        # Plain user message
        if item.get("role") == "user":
            messages.append({"role": "user", "content": item["content"]})
            i += 1
            continue

        # function_call_output → tool message
        if item.get("type") == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item["output"],
            })
            i += 1
            continue

        # Responses API output items (from model_dump): could be "message" or "function_call"
        # Collect contiguous assistant output into one assistant message
        if item.get("type") in ("message", "function_call"):
            assistant_content = ""
            tool_calls = []

            while i < len(input_items) and input_items[i].get("type") in ("message", "function_call"):
                cur = input_items[i]
                if cur["type"] == "message":
                    # Extract text from content blocks
                    for block in cur.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            assistant_content += block.get("text", "")
                elif cur["type"] == "function_call":
                    tool_calls.append({
                        "id": cur["call_id"],
                        "type": "function",
                        "function": {
                            "name": cur["name"],
                            "arguments": cur["arguments"],
                        },
                    })
                i += 1

            msg = {"role": "assistant"}
            if assistant_content:
                msg["content"] = assistant_content
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        # Fallback: skip unknown items
        i += 1

    # Add cache_control breakpoint on the last user or tool message
    for j in range(len(messages) - 1, -1, -1):
        if messages[j]["role"] in ("user", "tool"):
            messages[j]["cache_control"] = {"type": "ephemeral"}
            break

    return messages


def _completions_tools(tool_schemas):
    """Convert responses-format tool schemas to completions-format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["parameters"],
            },
        }
        for t in tool_schemas
    ]


class _FakeOutputItem:
    """Lightweight adapter so completions response items quack like responses output items."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name)

    def model_dump(self):
        def _unwrap(v):
            if isinstance(v, _FakeOutputItem):
                return v.model_dump()
            if isinstance(v, list):
                return [_unwrap(x) for x in v]
            return v
        return {k: _unwrap(v) for k, v in self._data.items()}


class _FakeUsage:
    def __init__(self, usage):
        self.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # Anthropic cache fields (litellm passes these through)
        self.cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # OpenAI-style cache details
        self.input_tokens_details = getattr(usage, "prompt_tokens_details", None)


class _FakeResponse:
    """Wraps a completions response to look like a responses response."""

    def __init__(self, response):
        self.model = response.model
        self.usage = _FakeUsage(response.usage) if response.usage else None
        self.output = self._build_output(response)

    def _build_output(self, response):
        choice = response.choices[0]
        msg = choice.message
        items = []

        # Reasoning content (e.g. Kimi K2.5 thinking blocks)
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            items.append(_FakeOutputItem({
                "type": "reasoning",
                "content": reasoning,
            }))

        # Text content → "message" item
        if msg.content:
            items.append(_FakeOutputItem({
                "type": "message",
                "content": [_FakeOutputItem({"type": "output_text", "text": msg.content})],
            }))

        # Tool calls → "function_call" items
        if msg.tool_calls:
            for tc in msg.tool_calls:
                items.append(_FakeOutputItem({
                    "type": "function_call",
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "call_id": tc.id,
                }))

        return items


def call_llm_completions(model, messages, tools, temperature=0.0,
                         max_output_tokens=None, timeout=None):
    """Call an LLM via LiteLLM Completions API with enxplicit caching."""
    kwargs = dict(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        num_retries=25,
        reasoning_effort="medium",
    )
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    # LiteLLM's together_ai/ provider drops tools= silently for MiniMax models.
    # Route via the openai-compatible endpoint against Together's base URL.
    if "MiniMax" in model:
        kwargs["api_base"] = "https://api.together.xyz/v1"
        kwargs["api_key"] = os.environ["TOGETHER_AI_API_KEY"]
    response = litellm.completion(**kwargs)
    return _FakeResponse(response)


@dataclass
class AgentResult:
    status: str  # submitted, limit exceeded, or error
    turns_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    boltz_calls: int = 0
    admet_calls: int = 0
    vina_calls: int = 0
    diffdock_calls: int = 0
    submitted_file: Optional[str] = None
    error: Optional[str] = None


def run_agent(task_yaml, task_dir, config: AgentConfig, container_id: str, output_dir: str, executor=None) -> AgentResult:
    """Run the agent loop for a single task.

    Args:
        task_yaml: Parsed task YAML dict (already obfuscated)
        task_dir: Path to the (obfuscated) task directory mounted in the container
        config: Agent configuration
        container_id: Docker container ID
        output_dir: Directory for logging output (trajectory.jsonl, transcript.md)

    Returns:
        AgentResult with status and metadata
    """
    logger = AgentLogger(output_dir, task_yaml.get("id", "unknown"))
    executor = executor or ToolExecutor(container_id, config)

    system_prompt = build_system_prompt(config)
    task_description = task_yaml.get("description", "No description provided.")

    logger.log_system_prompt(system_prompt)

    # Build initial input (system prompt passed via instructions parameter)
    initial_message = f"## Task\n\n{task_description}\n\nYour current working directory contains the task files. Start by examining them."
    input_items = [
        {"role": "user", "content": initial_message},
    ]
    logger.log_user_message(0, initial_message)

    REFUSAL_PHRASES = [
        "i cannot assist",
        "i can't assist",
        "i'm not able to",
        "i cannot help",
        "i can't help"
    ]

    turn = 0
    refusal_count = 0
    consecutive_no_tool = 0
    submit_guard = _SubmitRetryGuard()
    while turn < config.max_turns:
        turn += 1

        # Check token budget
        if logger.total_input_tokens + logger.total_output_tokens >= config.max_tokens:
            logger.log_summary("token_budget_exceeded", turn - 1)
            return AgentResult(
                status="token_budget_exceeded",
                turns_used=turn - 1,
                total_input_tokens=logger.total_input_tokens,
                total_output_tokens=logger.total_output_tokens,
                total_cache_read_tokens=logger.total_cache_read_tokens,
                total_cache_creation_tokens=logger.total_cache_creation_tokens,
                boltz_calls=executor.boltz_calls,
                admet_calls=executor.admet_calls,
                vina_calls=executor.vina_calls,
                diffdock_calls=executor.diffdock_calls,
            )

        # Call LLM
        try:
            if config.use_responses_api:
                response = call_llm(
                    model=config.model,
                    input_items=input_items,
                    tools=TOOL_SCHEMAS,
                    instructions=system_prompt,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    timeout=config.llm_timeout,
                )
            else:
                messages = _responses_input_to_messages(input_items, system_prompt)
                response = call_llm_completions(
                    model=config.model,
                    messages=messages,
                    tools=_completions_tools(TOOL_SCHEMAS),
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    timeout=config.llm_timeout,
                )
        except Exception as e:
            logger.log_summary("error", turn - 1)
            return AgentResult(
                status="error",
                turns_used=turn - 1,
                total_input_tokens=logger.total_input_tokens,
                total_output_tokens=logger.total_output_tokens,
                total_cache_read_tokens=logger.total_cache_read_tokens,
                total_cache_creation_tokens=logger.total_cache_creation_tokens,
                boltz_calls=executor.boltz_calls,
                admet_calls=executor.admet_calls,
                vina_calls=executor.vina_calls,
                diffdock_calls=executor.diffdock_calls,
                error=str(e),
            )

        logger.log_llm_call(turn, response)

        # Extract function calls from response output
        output_items = response.output
        function_calls = [item for item in output_items if item.type == "function_call"]

        # No tool calls: agent is talking without acting
        if not function_calls:
            consecutive_no_tool += 1
            input_items.extend([item.model_dump() for item in output_items])

            if (config.max_consecutive_no_tool is not None and
                    consecutive_no_tool >= config.max_consecutive_no_tool):
                logger.log_summary("no_tool_limit_exceeded", turn)
                return AgentResult(
                    status="no_tool_limit_exceeded",
                    turns_used=turn,
                    total_input_tokens=logger.total_input_tokens,
                    total_output_tokens=logger.total_output_tokens,
                    total_cache_read_tokens=logger.total_cache_read_tokens,
                    total_cache_creation_tokens=logger.total_cache_creation_tokens,
                    boltz_calls=executor.boltz_calls,
                    admet_calls=executor.admet_calls,
                    vina_calls=executor.vina_calls,
                    diffdock_calls=executor.diffdock_calls,
                )

            # Detect refusal — don't count this turn (up to max_refusals)
            text_parts = [item.text for item in output_items if hasattr(item, "text") and item.text]
            full_text = " ".join(text_parts).strip().lower()
            if any(phrase in full_text for phrase in REFUSAL_PHRASES) and refusal_count < config.max_refusals:
                refusal_count += 1
                turn -= 1
                logger.log_refusal(turn)

            nudge = "Please use your tools to work on the task, or call submit() when ready."
            input_items.append({
                "role": "user",
                "content": nudge,
            })
            logger.log_user_message(turn, nudge)
            continue

        # Process tool calls — append all output items first
        consecutive_no_tool = 0
        dumped_items = [item.model_dump() for item in output_items]
        # Sanitize malformed tool call arguments so they don't poison the next API call
        for d in dumped_items:
            if d.get("type") == "function_call":
                try:
                    json.loads(d["arguments"])
                except (json.JSONDecodeError, TypeError, KeyError):
                    d["arguments"] = "{}"
        input_items.extend(dumped_items)

        for fc in function_calls:
            tool_name = fc.name
            try:
                tool_args = json.loads(fc.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            submit_path = submit_guard.key(tool_args) if tool_name == "submit" else ""
            correction = None
            t0 = time.time()
            if tool_name == "submit" and submit_guard.blocked(submit_path):
                result_str = f"ERROR: Repeated submission blocked. {SUBMIT_CORRECTION}"
                extra = {"repeated_submit_blocked": True}
            else:
                if tool_name != "submit":
                    submit_guard.reset()
                result_str, extra = executor.dispatch(tool_name, tool_args)
                if tool_name == "submit":
                    if result_str.lstrip().startswith("ERROR"):
                        if submit_guard.record_failure(submit_path) == 2:
                            correction = SUBMIT_CORRECTION
                            result_str = f"{result_str}\n\n{correction}"
                    else:
                        submit_guard.reset()
            duration = time.time() - t0

            logger.log_tool_call(turn, tool_name, tool_args, result_str, duration, extra)

            # Update logger counters from executor
            logger.boltz_calls = executor.boltz_calls
            logger.admet_calls = executor.admet_calls
            logger.vina_calls = executor.vina_calls
            logger.diffdock_calls = executor.diffdock_calls

            # Append function call output to input
            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result_str,
            })
            if correction:
                input_items.append({"role": "user", "content": correction})
                logger.log_user_message(turn, correction)

            # Check for submission
            if tool_name == "submit" and not result_str.lstrip().startswith("ERROR"):
                submitted_file = extra.get("submitted_file", tool_args.get("file_path", ""))
                logger.log_submit(turn, submitted_file)
                logger.log_summary("submitted", turn)
                return AgentResult(
                    status="submitted",
                    turns_used=turn,
                    total_input_tokens=logger.total_input_tokens,
                    total_output_tokens=logger.total_output_tokens,
                    total_cache_read_tokens=logger.total_cache_read_tokens,
                    total_cache_creation_tokens=logger.total_cache_creation_tokens,
                    boltz_calls=executor.boltz_calls,
                    admet_calls=executor.admet_calls,
                    vina_calls=executor.vina_calls,
                    diffdock_calls=executor.diffdock_calls,
                    submitted_file=submitted_file,
                )

        # Warn the agent about remaining turns at key thresholds
        turns_left = config.max_turns - turn
        if turns_left in (50, 25, 10, 3):
            warning = f"You have {turns_left} turns remaining. Please plan accordingly."
            input_items.append({
                "role": "user",
                "content": warning,
            })
            logger.log_user_message(turn, warning)

    # Max turns exceeded
    logger.log_summary("max_turns_exceeded", config.max_turns)
    return AgentResult(
        status="max_turns_exceeded",
        turns_used=config.max_turns,
        total_input_tokens=logger.total_input_tokens,
        total_output_tokens=logger.total_output_tokens,
        total_cache_read_tokens=logger.total_cache_read_tokens,
        total_cache_creation_tokens=logger.total_cache_creation_tokens,
        boltz_calls=executor.boltz_calls,
        admet_calls=executor.admet_calls,
        vina_calls=executor.vina_calls,
        diffdock_calls=executor.diffdock_calls,
    )
