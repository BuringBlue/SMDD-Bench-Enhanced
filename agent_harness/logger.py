"""Structured JSONL logging and human-readable transcript for agent runs."""

import json
import logging
import os
import time
from datetime import datetime, timezone

_stdout_logger = logging.getLogger("smdd_agent")


class AgentLogger:
    def __init__(self, output_dir, task_id):
        os.makedirs(output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(output_dir, "trajectory.jsonl")
        self.transcript_path = os.path.join(output_dir, "transcript.md")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.boltz_calls = 0
        self.admet_calls = 0
        self.refusals = 0
        self._start_time = time.time()

        # Initialize trajectory with task id
        self._write_event({"event": "task_start", "task_id": task_id})

        # Initialize transcript
        with open(self.transcript_path, "w") as f:
            f.write(f"# Agent Transcript — {task_id}\n\nStarted: {datetime.now(timezone.utc).isoformat()}\n\n")

    def _write_event(self, event):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.trajectory_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        _stdout_logger.info(json.dumps(event, default=str))

    def _extract_cached_tokens(self, usage):
        """Extract cache read and creation tokens from the usage object.

        Handles both OpenAI format (input_tokens_details.cached_tokens) and
        Anthropic format (cache_read_input_tokens / cache_creation_input_tokens).
        """
        cache_read = 0
        cache_creation = 0
        if not usage:
            return cache_read, cache_creation

        # Anthropic format
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # OpenAI format: usage.input_tokens_details.cached_tokens
        if cache_read == 0:
            details = getattr(usage, "input_tokens_details", None)
            if details:
                cache_read = getattr(details, "cached_tokens", 0) or 0

        return cache_read, cache_creation

    def log_user_message(self, turn, content):
        self._write_event({
            "event": "user_message",
            "turn": turn,
            "content": content,
        })
        with open(self.transcript_path, "a") as f:
            f.write(f"**User:** {content}\n\n")

    def log_system_prompt(self, system_prompt):
        self._write_event({
            "event": "system_prompt",
            "content": system_prompt,
        })
        with open(self.transcript_path, "a") as f:
            f.write(f"## System Prompt\n\n{system_prompt}\n\n---\n\n")

    def log_llm_call(self, turn, response):
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        cache_read, cache_creation = self._extract_cached_tokens(usage)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read
        self.total_cache_creation_tokens += cache_creation

        tool_call_names = []
        assistant_text = ""
        reasoning_text = ""
        for item in response.output:
            if item.type == "function_call":
                tool_call_names.append(item.name)
            elif item.type == "reasoning":
                # litellm normalizes reasoning across providers inconsistently: some models
                # populate `summary` (blocks), some `content` (blocks or a raw string), some both.
                for block in (getattr(item, "summary", None) or []):
                    if getattr(block, "text", None):
                        reasoning_text += block.text
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    reasoning_text += content
                elif content:
                    for block in content:
                        if getattr(block, "text", None):
                            reasoning_text += block.text
            elif item.type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text" and content_block.text:
                        assistant_text += content_block.text

        event = {
            "event": "llm_call",
            "turn": turn,
            "model": response.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "tool_calls": tool_call_names,
            "assistant_text": assistant_text,
        }
        if reasoning_text:
            event["reasoning_content"] = reasoning_text
        self._write_event(event)

        # Transcript
        with open(self.transcript_path, "a") as f:
            f.write(f"## Turn {turn}\n\n")
            if reasoning_text:
                f.write(f"**Reasoning:** {reasoning_text}\n\n")
            if assistant_text:
                f.write(f"**Assistant:** {assistant_text}\n\n")

    def log_tool_call(self, turn, tool_name, tool_input, result, duration_s, extra=None):
        event = {
            "event": "tool_call",
            "turn": turn,
            "tool": tool_name,
            "input": tool_input,
            "output_truncated": (str(result)[:1000] + "\n\n... [middle truncated] ...\n\n" + str(result)[-1000:]) if len(str(result)) > 2000 else str(result),
            "duration_s": round(duration_s, 2),
        }
        if extra:
            event.update(extra)
        self._write_event(event)

        # Transcript
        reasoning = tool_input.get("reasoning", "")
        display_input = {k: v for k, v in tool_input.items() if k != "reasoning"}
        with open(self.transcript_path, "a") as f:
            f.write(f"**Tool Call:** {tool_name}\n")
            if reasoning:
                f.write(f"> **Reasoning:** {reasoning}\n\n")
            if tool_name == "python":
                f.write(f"```python\n{display_input.get('code', '')}\n```\n")
            elif tool_name == "bash":
                f.write(f"```bash\n{display_input.get('command', '')}\n```\n")
            else:
                f.write(f"```json\n{json.dumps(display_input, indent=4, default=str)}\n```\n")
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = result_str[:1000] + "\n\n... [middle truncated] ...\n\n" + result_str[-1000:]
            f.write(f"**Result:**\n```\n{result_str}\n```\n\n")

    def log_refusal(self, turn):
        self.refusals += 1
        self._write_event({"event": "refusal", "turn": turn, "total_refusals": self.refusals})

    def log_submit(self, turn, file_path, content_preview=""):
        self._write_event({
            "event": "submit",
            "turn": turn,
            "file": file_path,
            "content": content_preview[:1000],
        })
        with open(self.transcript_path, "a") as f:
            f.write(f"**Submitted:** `{file_path}`\n\n")

    def log_summary(self, status, total_turns):
        wall_time = time.time() - self._start_time
        summary = {
            "event": "summary",
            "status": status,
            "total_turns": total_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "boltz_calls": self.boltz_calls,
            "admet_calls": self.admet_calls,
            "wall_time_s": round(wall_time, 2),
        }
        self._write_event(summary)

        with open(self.transcript_path, "a") as f:
            f.write(f"\n---\n## Summary\n\n")
            f.write(f"- Status: {status}\n")
            f.write(f"- Turns: {total_turns}\n")
            f.write(f"- Input tokens: {self.total_input_tokens}\n")
            f.write(f"- Output tokens: {self.total_output_tokens}\n")
            f.write(f"- Cache read tokens: {self.total_cache_read_tokens}\n")
            f.write(f"- Cache creation tokens: {self.total_cache_creation_tokens}\n")
            f.write(f"- Boltz calls: {self.boltz_calls}\n")
            f.write(f"- ADMET calls: {self.admet_calls}\n")
            f.write(f"- Refusals: {self.refusals}\n")
            f.write(f"- Wall time: {wall_time:.1f}s\n")

        return summary
