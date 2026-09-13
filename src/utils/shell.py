import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str], timeout: float | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    if not isinstance(cmd, Sequence) or isinstance(cmd, (str, bytes)):
        raise TypeError("cmd must be a sequence of strings, for example ['ls', '-la']")
    if not cmd:
        raise ValueError("cmd cannot be empty")
    if any(not isinstance(part, str) or not part for part in cmd):
        raise ValueError("every command part must be a non-empty string")

    logger.debug("running command: %s (cwd=%s)", cmd, cwd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {cmd}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr output)"
        raise RuntimeError(f"command failed with exit code {result.returncode}: {cmd}\n{stderr}")

    return result
