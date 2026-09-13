import json
import logging
import shutil
from pathlib import Path

from utils.repo import parse_owner_repo
from utils.shell import run_command

logger = logging.getLogger(__name__)


def download_src_with_commit(repo_url: str, commit_sha: str, base_dir: Path) -> Path:
    owner, repo = parse_owner_repo(repo_url)

    repo_dir = Path(base_dir).expanduser().resolve().joinpath(owner, repo, commit_sha)
    if repo_dir.exists() and any(repo_dir.iterdir()):
        return repo_dir

    repo_dir.mkdir(parents=True, exist_ok=True)

    run_command(["git", "init"], cwd=repo_dir)
    run_command(["git", "remote", "add", "origin", repo_url], cwd=repo_dir)
    run_command(["git", "fetch", "--depth", "1", "origin", commit_sha], cwd=repo_dir)
    run_command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)

    shutil.rmtree(repo_dir / ".git")

    return repo_dir


def register_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "download",
        help="Download source repos listed in a benchmark JSONL file",
    )
    p.add_argument(
        "jsonl_path",
        nargs="?",
        default="benchmark/tasks/vex_bench.jsonl",
        help="Path to benchmark JSONL (default: benchmark/tasks/vex_bench.jsonl)",
    )
    p.add_argument(
        "--repos-dir",
        default="benchmark/repos",
        help="Directory to download repos into (default: benchmark/repos)",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    tasks = [json.loads(line) for line in Path(args.jsonl_path).read_text().splitlines() if line]
    repos_dir = Path(args.repos_dir)

    for task in tasks:
        task_id = task["task_id"]
        repo_url = task["repo_url"]
        commit_sha = task["commit_sha"]
        logger.info("[%s] %s @ %s", task_id, repo_url, commit_sha)
        try:
            repo_dir = download_src_with_commit(repo_url, commit_sha, repos_dir)
            logger.info("[%s] -> %s", task_id, repo_dir)
        except Exception as e:
            logger.error("[%s] FAILED: %s", task_id, e)
