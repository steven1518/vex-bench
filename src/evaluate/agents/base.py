import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass(frozen=True)
class RunContext:
    """Per-task runtime state passed from the eval loop to an agent.

    ``cwd`` points at the pristine source tree and MUST NOT be mutated.
    Agents that need to write into the source set up their own isolation
    (e.g. ``docker cp`` into a disposable container).
    """

    cwd: Path
    language: str
    timeout: int = 600
    artifacts_dir: Path | None = None


def run_in_container(
    image: str,
    args: list[str],
    *,
    entrypoint: str,
    src: Path,
    timeout: int,
    env_file: Path | None = None,
    mounts: Sequence[tuple[Path, str]] = (),
    workdir: str = "/work",
    artifacts_dir: Path | None = None,
    artifact_paths: Sequence[tuple[str, str]] = (),
) -> str:
    """Run a one-shot agent command in a disposable container; return stdout.

    Creates the container, copies ``src`` into ``workdir`` with ``docker cp``,
    runs it, and force-removes the container — writable layer included — when
    done. The workspace is never bind-mounted, so root-owned build artifacts
    die with the container instead of poisoning the host.

    ``mounts`` are extra read-only host->container bind mounts (e.g. a config
    file). Raises RuntimeError on any docker failure, a non-zero exit, or
    timeout.
    """
    create_cmd = ["docker", "create", "--workdir", workdir]
    if env_file is not None:
        create_cmd += ["--env-file", str(env_file)]
    for host_path, container_path in mounts:
        create_cmd += ["-v", f"{host_path}:{container_path}:ro"]
    create_cmd += ["--entrypoint", entrypoint, image, *args]

    try:
        created = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"docker create failed: {exc.stderr.strip() or '(no stderr)'}") from exc
    cid = created.stdout.strip()

    try:
        try:
            subprocess.run(
                ["docker", "cp", f"{src.resolve()}/.", f"{cid}:{workdir}"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"docker cp failed: {exc.stderr.strip() or '(no stderr)'}") from exc

        try:
            started = subprocess.run(
                ["docker", "start", "-a", cid],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"container timed out after {timeout}s{_captured_io(exc.stdout, exc.stderr)}"
            ) from exc

        if started.returncode != 0:
            raise RuntimeError(
                f"container exited {started.returncode}: "
                f"{started.stderr.strip() or '(no stderr)'}"
                f"{_captured_io(started.stdout, started.stderr)}"
            )
        return started.stdout
    finally:
        if artifacts_dir is not None:
            _copy_container_artifacts(cid, artifacts_dir, artifact_paths)
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)


def _copy_container_artifacts(
    cid: str, artifacts_dir: Path, artifact_paths: Sequence[tuple[str, str]]
) -> None:
    if not artifact_paths:
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for container_path, name in artifact_paths:
        dest = artifacts_dir / name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        subprocess.run(
            ["docker", "cp", f"{cid}:{container_path}", str(dest)],
            capture_output=True,
            check=False,
        )


def _captured_io(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    parts = []
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        text = _decode_capture(value).strip()
        if text:
            parts.append(f"\n{name} tail:\n{_tail(text)}")
    return "".join(parts)


def _decode_capture(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


class AgentResult(BaseModel):
    # Raw output from the agent. The format is agent-specific.
    # Each Agent's ``extract_text`` knows how to parse its own output.
    raw_output: str


class RunStats(BaseModel):
    """Trajectory-level stats parsed from an agent's raw_output.

    The five token fields are **disjoint buckets** — none is a subset of
    another — so a run's total billable tokens is simply their sum:
    ``input + cached_input + cache_write + output + reasoning``.

    Agents report tokens in their own native layout, which differs:
    Codex nests (``cached_input`` inside ``input``, ``reasoning`` inside
    ``output``) while opencode already splits them. Each agent's
    ``extract_stats`` is responsible for normalising onto this disjoint
    layout — e.g. Codex must subtract the nested counts.
    """

    steps: int | None = None
    completed: bool = False
    input_tokens: int | None = None  # fresh prompt tokens (NOT cache hits)
    cached_input_tokens: int | None = None  # prompt tokens served from cache (cache read)
    cache_write_tokens: int | None = None  # prompt tokens written to cache (cache creation)
    output_tokens: int | None = None  # completion tokens, EXCLUDING reasoning
    reasoning_tokens: int | None = None  # reasoning / thinking tokens


@dataclass(frozen=True)
class ParsedResult:
    path: Path
    raw: str
    text: str
    category: str
    reasoning: str | None
    stats: RunStats


class Agent(ABC):
    """A coding-agent CLI bound to a specific model.

    Concrete agents are responsible for the full execution — there is no
    separate Environment layer. Subclasses that need a sandbox run inside a
    disposable container via ``run_in_container``.
    """

    @abstractmethod
    def run(self, prompt: str, ctx: RunContext) -> AgentResult: ...

    @abstractmethod
    def extract_text(self, raw_output: str) -> str: ...

    @abstractmethod
    def result_filename(self) -> str:
        """Filename used by the run stage for newly written results."""
        ...

    @abstractmethod
    def extract_stats(self, raw_output: str) -> RunStats:
        """Parse trajectory stats from the agent-specific raw output."""
        ...

    @abstractmethod
    def load_result(self, run_dir: Path) -> ParsedResult | None:
        """Return a reusable parsed result from this run directory, if any."""
        ...

    @abstractmethod
    def has_result(self, run_dir: Path) -> bool:
        """Return whether this run directory contains a result file for this agent."""
        ...

    def save_report(self, result_dir: Path) -> None:
        """Optionally write a human-readable report. No-op by default."""
