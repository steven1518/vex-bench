from collections.abc import Callable

from evaluate.agents.base import Agent, AgentResult, ParsedResult, RunContext, RunStats
from evaluate.agents.claude import ClaudeAgent
from evaluate.agents.codex import CodexAgent
from evaluate.agents.opencode import OpenCodeAgent

# Each entry is a factory: ``(model) -> Agent``.
AGENTS: dict[str, Callable[..., Agent]] = {
    "codex": CodexAgent,
    "claude": ClaudeAgent,
    "opencode": OpenCodeAgent,
}

__all__ = ["AGENTS", "Agent", "AgentResult", "ParsedResult", "RunContext", "RunStats"]
