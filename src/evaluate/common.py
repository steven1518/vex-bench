import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator

from evaluate.agents import AGENTS, Agent
from evaluate.prompts import PROMPTS

BENCHMARKS: dict[str, Path] = {
    "vex_bench": Path("benchmark/tasks/vex_bench.jsonl"),
}

PROMPT_HASH_LEN = 12


@lru_cache(maxsize=None)
def _prompt_hash(text: str) -> str:
    """Stable short fingerprint of a prompt template (template, not rendered)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:PROMPT_HASH_LEN]


class ExperimentConfig(BaseModel):
    model: str
    agent: str
    benchmark: str
    prompt: str
    repeats: int = 1
    timeout: int = 600
    parallel: int = 1
    repos_dir: Path = Path("benchmark/repos")
    output_dir: Path = Path("results")

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, v: str) -> str:
        if v not in AGENTS:
            raise ValueError(f"Unknown agent: {v!r}. Available: {sorted(AGENTS)}")
        return v

    @field_validator("benchmark")
    @classmethod
    def _validate_benchmark(cls, v: str) -> str:
        if v not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {v!r}. Available: {sorted(BENCHMARKS)}")
        return v

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, v: str) -> str:
        if v not in PROMPTS:
            raise ValueError(f"Unknown prompt: {v!r}. Available: {sorted(PROMPTS)}")
        return v

    @field_validator("repeats")
    @classmethod
    def _validate_repeats(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"repeats must be >= 1, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"timeout must be >= 1, got {v}")
        return v

    @field_validator("parallel")
    @classmethod
    def _validate_parallel(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"parallel must be >= 1, got {v}")
        return v

    @property
    def benchmark_path(self) -> Path:
        return BENCHMARKS[self.benchmark]

    @property
    def prompt_text(self) -> str:
        return PROMPTS[self.prompt]

    @property
    def prompt_hash(self) -> str:
        return _prompt_hash(self.prompt_text)

    @property
    def run_dir(self) -> Path:
        """Where this experiment's per-task outputs live."""
        return self.output_dir / self.prompt_hash / self.agent / self.model

    @property
    def benchmark_dir(self) -> Path:
        """Per-benchmark scope inside this run (parsed.jsonl, future metrics.json, ...)."""
        return self.run_dir / self.benchmark


def make_agent(config: ExperimentConfig) -> Agent:
    return AGENTS[config.agent](model=config.model)


def load_tasks(config: ExperimentConfig) -> list[dict]:
    return [
        json.loads(line)
        for line in config.benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Register the CLI flags that map onto ExperimentConfig."""
    parser.add_argument("--model", default="kimi-k2.6", help="Model identifier")
    parser.add_argument(
        "--agent",
        default="opencode",
        choices=sorted(AGENTS),
        help="Agent runner name (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark",
        default="vex_bench",
        choices=sorted(BENCHMARKS),
        help="Benchmark name (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="vuln",
        choices=sorted(PROMPTS),
        help="Prompt name (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent agent runs per task (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-run timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max concurrent agent runs (default: %(default)s, i.e. sequential)",
    )
    parser.add_argument(
        "--repos-dir",
        default="benchmark/repos",
        help="Directory containing source repos (default: benchmark/repos)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for experiment outputs (default: results)",
    )


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        model=args.model,
        agent=args.agent,
        benchmark=args.benchmark,
        prompt=args.prompt,
        repeats=args.repeats,
        timeout=args.timeout,
        parallel=args.parallel,
        repos_dir=Path(args.repos_dir),
        output_dir=Path(args.output_dir),
    )
