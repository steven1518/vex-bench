import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluate.agents import Agent, RunContext
from evaluate.common import (
    ExperimentConfig,
    add_experiment_args,
    config_from_args,
    load_tasks,
    make_agent,
)
from utils.repo import parse_owner_repo

logger = logging.getLogger(__name__)


def _run_one(task: dict, rep: int, *, config: ExperimentConfig, agent: Agent) -> None:
    """Execute one (task, repeat) pair; write the agent's raw result on success."""
    task_id = task["task_id"]
    run_id = f"run_{rep:03d}"
    tag = f"[{task_id}/{run_id}]"

    language = task.get("metadata", {}).get("language")
    if not language:
        logger.warning("%s [SKIP] Missing metadata.language", tag)
        return

    owner, repo = parse_owner_repo(task["repo_url"])
    src_cwd = config.repos_dir / owner / repo / task["commit_sha"]
    if not src_cwd.exists():
        logger.warning("%s [SKIP] Source dir not found: %s", tag, src_cwd)
        return

    run_dir_per_rep = config.run_dir / task_id / run_id
    error_path = run_dir_per_rep / "error.txt"
    cached = agent.load_result(run_dir_per_rep)
    if cached is not None:
        error_path.unlink(missing_ok=True)
        logger.info("%s [CACHE] %s", tag, cached.path)
        return
    if agent.has_result(run_dir_per_rep):
        logger.info("%s [RE-RUN] cached result unparseable: %s", tag, run_dir_per_rep)

    run_dir_per_rep.mkdir(parents=True, exist_ok=True)
    prompt = config.prompt_text.replace("{cve_id}", task["cve_id"])
    logger.info("%s [RUN] src=%s lang=%s", tag, src_cwd, language)
    ctx = RunContext(
        cwd=src_cwd,
        language=language,
        timeout=config.timeout,
        artifacts_dir=run_dir_per_rep / "artifacts",
    )
    try:
        agent_result = agent.run(prompt, ctx)
    except RuntimeError as exc:
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        logger.warning("%s [ERROR] %s", tag, exc)
        return

    error_path.unlink(missing_ok=True)
    result_path = run_dir_per_rep / agent.result_filename()
    result_path.write_text(agent_result.raw_output, encoding="utf-8")
    agent.save_report(run_dir_per_rep)


def run(config: ExperimentConfig) -> None:
    """Execute the agent over every task and persist raw output under config.run_dir.

    Writes one raw result file per task and repeat; each agent chooses its
    filename via ``result_filename``.
    Parsing and metrics are downstream stages and live in separate modules.
    """
    agent = make_agent(config)
    tasks = load_tasks(config)
    logger.info("Loaded %d tasks from %s", len(tasks), config.benchmark_path)

    prompt_dir = config.output_dir / config.prompt_hash
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "prompt.txt"
    if not prompt_file.exists():
        prompt_file.write_text(config.prompt_text, encoding="utf-8")

    logger.info(
        "prompt=%s hash=%s parallel=%d", config.prompt, config.prompt_hash, config.parallel
    )

    jobs = [(task, rep) for task in tasks for rep in range(1, config.repeats + 1)]
    total = len(jobs)

    with ThreadPoolExecutor(max_workers=config.parallel) as ex:
        futs = {
            ex.submit(_run_one, t, r, config=config, agent=agent): (t["task_id"], r)
            for t, r in jobs
        }
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                task_id, rep = futs[fut]
                try:
                    fut.result()
                except Exception:
                    logger.exception("[%s/run_%03d] unexpected failure", task_id, rep)
                logger.info("progress %d/%d", i, total)
        except KeyboardInterrupt:
            logger.warning("interrupted — cancelling pending; waiting for in-flight to clean up")
            ex.shutdown(wait=True, cancel_futures=True)
            raise


def register_parser(subparsers) -> None:
    p = subparsers.add_parser("run", help="Run vulnerability exploitability evaluation")
    add_experiment_args(p)
    p.set_defaults(func=lambda args: run(config_from_args(args)))
