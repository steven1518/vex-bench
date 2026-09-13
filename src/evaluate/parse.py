import json
import logging
from pathlib import Path

from evaluate.agents import Agent, RunStats
from evaluate.common import (
    ExperimentConfig,
    add_experiment_args,
    config_from_args,
    load_tasks,
    make_agent,
)
from evaluate.pricing import cost_usd
from evaluate.result_parser import normalize_ground_truth, to_binary

logger = logging.getLogger(__name__)


def parse(config: ExperimentConfig) -> None:
    """Parse runs matching ``config`` into ``<benchmark_dir>/parsed.jsonl``.

    Driven by the benchmark's tasks, so the run dir can be reused across
    benchmark variants. Any (task, run) that can't yield a category becomes a
    row with ``predicted=None`` and a ``status`` of ``parse_fail`` /
    ``no_result`` / ``no_dir`` (``ok`` otherwise) — metrics counts these as
    failures, not skips. The only hard error is the run dir not existing.
    """
    agent = make_agent(config)
    tasks = load_tasks(config)

    if not config.run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {config.run_dir}")

    logger.info(
        "Parsing benchmark=%s tasks=%d repeats=%d under %s",
        config.benchmark,
        len(tasks),
        config.repeats,
        config.run_dir,
    )

    rows = [
        _parse_run(agent, task, config.run_dir, rep, config.model)
        for task in tasks
        for rep in range(1, config.repeats + 1)
    ]

    config.benchmark_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = config.benchmark_dir / "parsed.jsonl"
    with parsed_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), parsed_path)


def _parse_run(agent: Agent, task: dict, run_dir: Path, rep: int, model: str) -> dict:
    task_id = task["task_id"]
    run_id = f"run_{rep:03d}"
    gt_raw = task["ground_truth"]
    row = {
        "task_id": task_id,
        "run_id": run_id,
        "cve_id": task["cve_id"],
        "ground_truth": normalize_ground_truth(gt_raw) if gt_raw is not None else None,
        "ground_truth_category": task.get("ground_truth_category"),
        "predicted": None,
        "predicted_category": None,
        "reasoning": None,
        "status": "ok",
        **RunStats().model_dump(),
        "cost_usd": None,
    }

    run_dir_per_rep = run_dir / task_id / run_id
    parsed = agent.load_result(run_dir_per_rep)
    if parsed is None:
        status = "no_dir" if not (run_dir / task_id).exists() else "no_result"
        if agent.has_result(run_dir_per_rep):
            status = "parse_fail"
        logger.warning("[%s] %s/%s", status.upper().replace("_", "-"), task_id, run_id)
        return {**row, "status": status}

    row.update(parsed.stats.model_dump())
    row["cost_usd"] = cost_usd(parsed.stats, model)

    return {
        **row,
        "predicted": to_binary(parsed.category),
        "predicted_category": parsed.category,
        "reasoning": parsed.reasoning,
    }


def register_parser(subparsers) -> None:
    p = subparsers.add_parser("parse", help="Parse recorded agent results into parsed.jsonl")
    add_experiment_args(p)
    p.set_defaults(func=lambda args: parse(config_from_args(args)))
