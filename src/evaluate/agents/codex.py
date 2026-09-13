import json
import logging
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

HOST_ENV_FILE = Path("env/codex/codex.env")
HOST_CONFIG = Path("env/codex/config.toml")
CONTAINER_CONFIG = "/tmp/.codex/config.toml"

class CodexAgent(Agent):
    """OpenAI Codex CLI, executed inside ``vex-bench-<language>-codex:latest``.

    Each run uses an ephemeral container: source is copied in via ``docker cp``
    and the container (with its writable layer) is removed afterwards. No
    bind mount, so root-owned build artifacts vanish with the container
    instead of poisoning the host filesystem.
    """

    def __init__(self, model: str):
        self.model = model

    def run(self, prompt: str, ctx: RunContext) -> AgentResult:
        env_file = HOST_ENV_FILE.resolve()
        if not env_file.exists():
            raise RuntimeError(f"Env file not found: {env_file}")
        config_file = HOST_CONFIG.resolve()
        if not config_file.exists():
            raise RuntimeError(f"Config file not found: {config_file}")

        image = f"vex-bench-{ctx.language}-codex:latest"
        logger.debug("running codex image=%s src=%s", image, ctx.cwd)
        stdout = run_in_container(
            image,
            [
                "exec",
                "--sandbox",
                "danger-full-access",
                "--json",
                "--skip-git-repo-check",
                "--model",
                self.model,
                prompt,
            ],
            entrypoint="codex",
            src=ctx.cwd,
            timeout=ctx.timeout,
            env_file=env_file,
            mounts=[(config_file, CONTAINER_CONFIG)],
        )
        return AgentResult(raw_output=stdout)

    def result_filename(self) -> str:
        return "result.jsonl"

    def extract_text(self, jsonl_text: str) -> str:
        """Last agent_message text from Codex NDJSON events."""
        events = _parse_jsonl(jsonl_text)
        texts = [t for e in events if (t := _agent_message_text(e))]
        return texts[-1].strip() if texts else ""

    def extract_stats(self, jsonl_text: str) -> RunStats:
        """Steps = every ``item.completed`` (agent_message / command_execution / web_search);
        completed = a ``turn.completed`` event was emitted before the stream ended;
        tokens come from ``turn.completed.usage``.

        Codex reports nested counts (Responses API convention):
        ``cached_input_tokens`` is a subset of ``input_tokens`` and
        ``reasoning_output_tokens`` is a subset of ``output_tokens``. RunStats
        wants disjoint buckets, so the nested counts are subtracted out here.
        Codex/OpenAI never reports cache *writes*, so ``cache_write_tokens``
        stays unset."""
        events = _parse_jsonl(jsonl_text)
        steps = sum(1 for e in events if e.get("type") == "item.completed")
        completed = any(e.get("type") == "turn.completed" for e in events)
        usage = next(
            (
                e["usage"]
                for e in events
                if e.get("type") == "turn.completed" and isinstance(e.get("usage"), dict)
            ),
            {},
        )
        cached = usage.get("cached_input_tokens")
        reasoning = usage.get("reasoning_output_tokens")
        return RunStats(
            steps=steps,
            completed=completed,
            input_tokens=_minus_nested(usage.get("input_tokens"), cached),
            cached_input_tokens=cached,
            output_tokens=_minus_nested(usage.get("output_tokens"), reasoning),
            reasoning_tokens=reasoning,
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


def _minus_nested(total: int | None, nested: int | None) -> int | None:
    """Subtract a nested sub-count from its total; keep None if total is absent."""
    if total is None:
        return None
    return max(0, total - (nested or 0))


def _agent_message_text(event: dict) -> str | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text if isinstance(text, str) else None


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
