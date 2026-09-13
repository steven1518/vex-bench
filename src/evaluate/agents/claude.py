import json
import logging
from collections import Counter
from pathlib import Path

from evaluate.agents.base import (
    Agent,
    AgentResult,
    ParsedResult,
    RunContext,
    RunStats,
    run_in_container,
)
from evaluate.result_parser import parse_category

logger = logging.getLogger(__name__)

HOST_ENV_FILE = Path("env/claude/claude.env")

# Per-model reasoning effort. Both values match the v2.1.117 Claude Code
# defaults — pinning explicitly so the experiment stays reproducible if
# the upstream default changes, and so a Foundry deployment-name pattern
# miss can't silently disable effort. Models not listed fall through to
# whatever Claude Code's default is for them.
#
# Opus 4.6 only supports low/medium/high/max (no xhigh — that level
# exists on Opus 4.7 only).
EFFORT_BY_MODEL = {
    "claude-sonnet-4-6": "high",
    "claude-opus-4-6": "high",
}


class ClaudeAgent(Agent):
    """Claude Code CLI, executed inside ``vex-bench-<language>-claude:latest``.

    Same ephemeral-container model as codex/opencode: source is copied in via
    ``docker cp`` and the container is destroyed afterwards. Foundry (Azure
    AI Foundry) credentials and deployment-name overrides come from
    ``env/claude/claude.env`` via ``--env-file``; Claude Code reads all
    provider config from env vars, so there is no separate config file to
    mount.
    """

    def __init__(self, model: str):
        self.model = model

    def run(self, prompt: str, ctx: RunContext) -> AgentResult:
        env_file = HOST_ENV_FILE.resolve()
        if not env_file.exists():
            raise RuntimeError(f"Env file not found: {env_file}")

        image = f"vex-bench-{ctx.language}-claude:latest"
        args = [
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--dangerously-skip-permissions",
        ]
        effort = EFFORT_BY_MODEL.get(self.model)
        if effort is not None:
            args += ["--effort", effort]
        args.append(prompt)

        logger.debug(
            "running claude model=%s effort=%s image=%s src=%s",
            self.model,
            effort,
            image,
            ctx.cwd,
        )
        stdout = run_in_container(
            image,
            args,
            entrypoint="claude",
            src=ctx.cwd,
            timeout=ctx.timeout,
            env_file=env_file,
        )
        return AgentResult(raw_output=stdout)

    def result_filename(self) -> str:
        return "result.jsonl"

    def extract_text(self, jsonl_text: str) -> str:
        """Final answer: prefer terminal ``result`` event, else last assistant message."""
        events = _parse_jsonl(jsonl_text)

        for e in reversed(events):
            if e.get("type") == "result":
                final = e.get("result")
                if isinstance(final, str) and final.strip():
                    return final.strip()
                break

        for e in reversed(events):
            if e.get("type") == "assistant":
                msg = e.get("message", {})
                content = msg.get("content", [])
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                joined = "\n".join(t for t in texts if t).strip()
                if joined:
                    return joined
        return ""

    def extract_stats(self, jsonl_text: str) -> RunStats:
        """Steps = ``result.num_turns``; completed = result event with
        ``subtype=="success"`` and ``is_error==false``; tokens come from
        ``result.modelUsage`` summed across all models that ran in the turn.

        The flat ``result.usage`` object has the same field names but
        ``input_tokens`` / ``output_tokens`` are unreliable streaming
        placeholders — ``modelUsage`` is the authoritative source matching
        ``total_cost_usd`` (see anthropics/claude-agent-sdk-typescript#112).
        Token names there are camelCase. Anthropic does not report
        thinking/reasoning tokens separately — they bill as output — so
        ``reasoning_tokens`` stays unset.
        """
        events = _parse_jsonl(jsonl_text)
        result_event = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if result_event is None:
            return RunStats()

        completed = result_event.get("subtype") == "success" and not result_event.get(
            "is_error", False
        )
        steps = result_event.get("num_turns")

        per_model = [
            m for m in (result_event.get("modelUsage") or {}).values() if isinstance(m, dict)
        ]
        if not per_model:
            return RunStats(steps=steps, completed=completed)
        return RunStats(
            steps=steps,
            completed=completed,
            input_tokens=sum(m.get("inputTokens", 0) for m in per_model),
            cached_input_tokens=sum(m.get("cacheReadInputTokens", 0) for m in per_model),
            cache_write_tokens=sum(m.get("cacheCreationInputTokens", 0) for m in per_model),
            output_tokens=sum(m.get("outputTokens", 0) for m in per_model),
        )

    def load_result(self, run_dir: Path) -> ParsedResult | None:
        path = run_dir / "result.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            text = self.extract_text(raw)
            category, reasoning = parse_category(text)
        except Exception:
            return None
        if category is None:
            return None
        try:
            stats = self.extract_stats(raw)
        except Exception:
            stats = RunStats()
        return ParsedResult(
            path=path,
            raw=raw,
            text=text,
            category=category,
            reasoning=reasoning,
            stats=stats,
        )

    def has_result(self, run_dir: Path) -> bool:
        return (run_dir / "result.jsonl").exists()

    def save_report(self, result_dir: Path) -> None:
        result_jsonl_path = result_dir / "result.jsonl"
        if result_jsonl_path.exists():
            md = self.jsonl_to_markdown(result_jsonl_path)
            (result_dir / "report.md").write_text(md, encoding="utf-8")

    def jsonl_to_markdown(self, jsonl_path: Path) -> str:
        """Convert a saved stream-json run log to a readable Markdown report."""
        events = _parse_jsonl(jsonl_path.read_text(encoding="utf-8"))
        result_event = next((e for e in reversed(events) if e.get("type") == "result"), None)

        tool_blocks: list[dict] = []
        for e in events:
            if e.get("type") != "assistant":
                continue
            for block in e.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_blocks.append(block)
        tool_counter = Counter(b.get("name", "unknown") for b in tool_blocks)

        duration_s = None
        num_turns = None
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        final_output = "_(No output text found)_"
        if result_event is not None:
            duration_ms = result_event.get("duration_ms")
            if isinstance(duration_ms, int | float):
                duration_s = duration_ms / 1000
            num_turns = result_event.get("num_turns")
            for m in (result_event.get("modelUsage") or {}).values():
                if not isinstance(m, dict):
                    continue
                tokens["input"] += m.get("inputTokens", 0)
                tokens["output"] += m.get("outputTokens", 0)
                tokens["cache_read"] += m.get("cacheReadInputTokens", 0)
                tokens["cache_write"] += m.get("cacheCreationInputTokens", 0)
            final = result_event.get("result")
            if isinstance(final, str) and final.strip():
                final_output = final.strip()

        lines = ["# Run Report", "", "## Stats", ""]
        if duration_s is not None:
            lines.append(f"- **Duration**: {duration_s:.1f}s")
        if num_turns is not None:
            lines.append(f"- **Turns**: {num_turns}")
        lines += [
            f"- **Tool calls**: {len(tool_blocks)}",
            "- **Tokens** — "
            + f"in: {tokens['input']:,} · out: {tokens['output']:,}"
            + (f" · cache_r: {tokens['cache_read']:,}" if tokens["cache_read"] else "")
            + (f" · cache_w: {tokens['cache_write']:,}" if tokens["cache_write"] else ""),
            "",
            "## Final Answer",
            "",
            final_output,
            "",
            "## Tool Calls",
            "",
        ]
        if tool_counter:
            lines += [
                "| Tool | Count |",
                "|---|---:|",
                *[
                    f"| `{t}` | {c} |"
                    for t, c in sorted(tool_counter.items(), key=lambda x: (-x[1], x[0]))
                ],
                "",
                "### Details",
                "",
            ]
            for block in tool_blocks:
                name = block.get("name", "unknown")
                summary = _summarize_tool_input(name, block.get("input", {}))
                lines.append(f"- **`{name}`** {summary}")
            lines.append("")
        else:
            lines += ["_No tool calls found._", ""]

        return "\n".join(lines)


def _summarize_tool_input(tool: str, inp: dict) -> str:
    if tool == "Bash":
        cmd = inp.get("command", "")
        return f"— `{cmd[:80]}{'…' if len(cmd) > 80 else ''}`"
    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        return f"— `{inp.get('file_path', '')}`"
    if tool == "Grep":
        parts = [f"`{inp.get('pattern', '')}`"]
        if "path" in inp:
            parts.append(f"in `{inp['path']}`")
        return "— " + " ".join(parts)
    if tool == "Glob":
        return f"— `{inp.get('pattern', '')}`"
    if tool == "WebFetch":
        return f"— `{inp.get('url', '')}`"
    if tool == "WebSearch":
        return f"— `{inp.get('query', '')}`"
    summary = ", ".join(f"{k}={v!r}" for k, v in list(inp.items())[:2])
    return f"— {summary}" if summary else ""


def _parse_jsonl(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events
