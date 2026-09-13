import json
import logging
import statistics
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from evaluate.common import ExperimentConfig, add_experiment_args, config_from_args
from evaluate.result_parser import BINARY_LABELS, CATEGORY_ORDER

logger = logging.getLogger(__name__)

FAILED = "failed"

# The five disjoint RunStats token buckets, summed and reported as-is.
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def compute_metrics(rows: list[dict]) -> dict:
    by_run = _group_by_run(rows)
    if not by_run:
        return {
            "binary": _empty_binary_agg(),
            "multiclass": _empty_multi_agg(),
            "completion": _empty_completion_agg(),
            "tokens": _empty_tokens_agg(),
            "cost": _empty_cost_agg(),
            "n_runs": 0,
        }

    binary_per_run = [_binary_one_run(rs) for rs in by_run.values()]
    multi_per_run = [_multiclass_one_run(rs) for rs in by_run.values()]
    completion_per_run = [_completion_one_run(rs) for rs in by_run.values()]
    return {
        "binary": _agg_binary(binary_per_run),
        "multiclass": _agg_multiclass(multi_per_run),
        "completion": _agg_completion(completion_per_run),
        "tokens": _agg_tokens(rows),
        "cost": _agg_cost(rows),
        "n_runs": len(by_run),
    }


def metrics(config: ExperimentConfig) -> None:
    """Read ``parsed.jsonl``, compute metrics, write ``metrics.json``."""
    parsed_path = config.benchmark_dir / "parsed.jsonl"
    if not parsed_path.exists():
        raise FileNotFoundError(f"parsed.jsonl not found: {parsed_path}")

    rows = [
        json.loads(line)
        for line in parsed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d rows from %s", len(rows), parsed_path)

    out = {
        "config": {
            "benchmark": config.benchmark,
            "agent": config.agent,
            "model": config.model,
            "prompt_hash": config.prompt_hash,
            "repeats": config.repeats,
        },
        **compute_metrics(rows),
    }

    metrics_path = config.benchmark_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", metrics_path)
    _print_report(out)


# ---------- per-run computation ----------


def _binary_one_run(rows: list[dict]) -> dict:
    """Binary metrics for one run. A row with a ground truth but no prediction
    is scored as ``FAILED``: wrong for accuracy, a miss (FN) for recall, and
    never a false positive — so precision is unaffected by failures."""
    pairs = [(r.get("ground_truth"), r.get("predicted")) for r in rows if r.get("ground_truth")]
    if not pairs:
        return _empty_binary_one_run()

    y_true = [gt for gt, _ in pairs]
    y_pred = [pred if pred is not None else FAILED for _, pred in pairs]
    labels = [*BINARY_LABELS, FAILED]
    pos = labels.index("exploitable")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # cm rows/cols = [not_exploitable, exploitable, failed]; the failed row is
    # always empty (no ground truth is ever "failed").
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    failed = int(cm[0][2] + cm[1][2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec[pos]),
        "recall": float(rec[pos]),
        "f1": float(f1[pos]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "failed": failed,
    }


def _multiclass_one_run(rows: list[dict]) -> dict:
    """Multiclass metrics for one run. A row with a ground-truth category but no
    prediction is scored as ``FAILED``: wrong for accuracy and a miss for the
    true class's recall, but never a false positive for any real class."""
    pairs = [
        (r.get("ground_truth_category"), r.get("predicted_category"))
        for r in rows
        if r.get("ground_truth_category")
    ]
    if not pairs:
        return _empty_multi_one_run()

    y_true = [gt for gt, _ in pairs]
    y_pred = [pred if pred is not None else FAILED for _, pred in pairs]
    gt_labels = _sorted_categories(set(y_true))
    failed = sum(1 for p in y_pred if p == FAILED)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=gt_labels, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=gt_labels, average="weighted", zero_division=0
    )
    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=gt_labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[*CATEGORY_ORDER, FAILED])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
        "per_class": {
            label: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(gt_labels)
        },
        "confusion_matrix": cm.tolist(),
        "failed": failed,
    }


def _completion_one_run(rows: list[dict]) -> dict:
    """How often the agent produced a parseable answer in one run."""
    total = len(rows)
    answered = sum(1 for r in rows if r.get("predicted") is not None)
    status_counts = Counter(r.get("status", "unknown") for r in rows)
    return {
        "total": total,
        "answered": answered,
        "completion_rate": answered / total if total else None,
        "status_counts": dict(status_counts),
    }


# ---------- aggregation across runs ----------


def _agg_binary(per_run: list[dict]) -> dict:
    scalar_keys = ["accuracy", "precision", "recall", "f1"]
    count_keys = ["tp", "fp", "tn", "fn", "failed"]
    return {
        **{k: _agg_scalars(m[k] for m in per_run) for k in scalar_keys},
        "totals": {k: sum(m[k] for m in per_run) for k in count_keys},
    }


def _agg_multiclass(per_run: list[dict]) -> dict:
    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    scalars = {k: _agg_scalars(m.get(k) for m in per_run) for k in scalar_keys}

    classes_seen: set[str] = set()
    for m in per_run:
        classes_seen.update(m.get("per_class", {}).keys())
    per_class_agg = {}
    for cls in _sorted_categories(classes_seen):
        per_class_agg[cls] = {
            field: _agg_scalars(m.get("per_class", {}).get(cls, {}).get(field) for m in per_run)
            for field in ("precision", "recall", "f1")
        }
        per_class_agg[cls]["support"] = sum(
            m.get("per_class", {}).get(cls, {}).get("support", 0) for m in per_run
        )

    cms = [np.array(m["confusion_matrix"]) for m in per_run if m.get("confusion_matrix")]
    cm_sum = np.sum(cms, axis=0).tolist() if cms else None

    return {
        **scalars,
        "per_class": per_class_agg,
        "confusion_matrix": {"labels": [*CATEGORY_ORDER, FAILED], "matrix": cm_sum},
        "failed_total": sum(m.get("failed", 0) for m in per_run),
    }


def _agg_completion(per_run: list[dict]) -> dict:
    statuses: set[str] = set()
    for m in per_run:
        statuses.update(m["status_counts"])
    return {
        "completion_rate": _agg_scalars(m["completion_rate"] for m in per_run),
        "total": sum(m["total"] for m in per_run),
        "answered": sum(m["answered"] for m in per_run),
        "status_counts": {
            s: sum(m["status_counts"].get(s, 0) for m in per_run) for s in sorted(statuses)
        },
    }


def _agg_tokens(rows: list[dict]) -> dict:
    """Token usage across every (task, repeat) case.

    ``per_case`` is the mean±std of one case's usage — the natural unit, and
    comparable across benchmarks of different sizes. ``totals`` is the grand
    sum over the whole experiment. Both cover each disjoint bucket plus their
    ``total``. Rows that produced no result (all buckets None) are skipped so
    they don't drag the per-case mean toward zero."""
    keys = (*TOKEN_FIELDS, "total")
    cases = []
    for r in rows:
        if all(r.get(f) is None for f in TOKEN_FIELDS):
            continue
        vals = {f: r.get(f) or 0 for f in TOKEN_FIELDS}
        vals["total"] = sum(vals.values())
        cases.append(vals)
    return {
        "per_case": {k: _agg_scalars(c[k] for c in cases) for k in keys},
        "totals": {k: sum(c[k] for c in cases) for k in keys},
        "cases": len(cases),
    }


def _agg_cost(rows: list[dict]) -> dict:
    """USD cost across every (task, repeat) case.

    ``per_case_usd`` is the mean±std cost of one case — the comparable unit;
    ``total_cost_usd`` is the whole experiment's spend. ``cost_usd`` is None
    for rows whose model is absent from ``PRICING`` (or that produced no
    tokens); ``priced`` / ``total`` expose that coverage so an incomplete
    price table is visible rather than silently understating cost."""
    priced = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    return {
        "per_case_usd": _agg_scalars(priced),
        "total_cost_usd": sum(priced),
        "priced": len(priced),
        "total": len(rows),
    }


def _agg_scalars(values) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


# ---------- helpers ----------


def _group_by_run(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["run_id"]].append(r)
    return dict(groups)


def _sorted_categories(cats) -> list[str]:
    return sorted(cats, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99)


def _empty_binary_one_run() -> dict:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "failed": 0,
    }


def _empty_binary_agg() -> dict:
    return {
        **{k: {"mean": None, "std": None} for k in ("accuracy", "precision", "recall", "f1")},
        "totals": {k: 0 for k in ("tp", "fp", "tn", "fn", "failed")},
    }


def _empty_multi_agg() -> dict:
    scalars = {
        k: {"mean": None, "std": None}
        for k in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        )
    }
    return {
        **scalars,
        "per_class": {},
        "confusion_matrix": {"labels": [*CATEGORY_ORDER, FAILED], "matrix": None},
        "failed_total": 0,
    }


def _empty_multi_one_run() -> dict:
    return {
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
        "per_class": {},
        "confusion_matrix": None,
        "failed": 0,
    }


def _empty_completion_agg() -> dict:
    return {
        "completion_rate": {"mean": None, "std": None},
        "total": 0,
        "answered": 0,
        "status_counts": {},
    }


def _empty_tokens_agg() -> dict:
    keys = (*TOKEN_FIELDS, "total")
    return {
        "per_case": {k: {"mean": None, "std": None} for k in keys},
        "totals": {k: 0 for k in keys},
        "cases": 0,
    }


def _empty_cost_agg() -> dict:
    return {
        "per_case_usd": {"mean": None, "std": None},
        "total_cost_usd": 0.0,
        "priced": 0,
        "total": 0,
    }


# ---------- reporting ----------


def _print_report(out: dict) -> None:
    binary = out["binary"]
    multi = out["multiclass"]
    comp = out["completion"]
    tokens = out["tokens"]
    cost = out["cost"]

    logger.info("\n=== Completion ===")
    logger.info(
        "  completion_rate %s  (answered %d / %d)",
        _fmt_mean_std(comp["completion_rate"]),
        comp["answered"],
        comp["total"],
    )
    if comp["status_counts"]:
        logger.info(
            "  status     %s",
            "  ".join(f"{s}={n}" for s, n in comp["status_counts"].items()),
        )

    logger.info("\n=== Tokens (per case, %d cases) ===", tokens["cases"])
    tot = tokens["totals"]
    per = tokens["per_case"]
    for f in (*TOKEN_FIELDS, "total"):
        logger.info(
            "  %-19s per-case %-21s total %s",
            f,
            _fmt_int_mean_std(per[f]),
            f"{tot[f]:,}",
        )

    logger.info("\n=== Cost (USD) ===")
    pcc = cost["per_case_usd"]
    if pcc["mean"] is None:
        logger.info("  no priced cases (model missing from evaluate.pricing.PRICING)")
    else:
        logger.info(
            "  per-case   %s    total $%.4f",
            _fmt_cost_mean_std(pcc),
            cost["total_cost_usd"],
        )
        if cost["priced"] < cost["total"]:
            logger.info(
                "  WARNING: only %d / %d cases priced (incomplete PRICING table)",
                cost["priced"],
                cost["total"],
            )

    logger.info("\n=== Binary (per-run-then-mean, n_runs=%d) ===", out["n_runs"])
    for k in ("accuracy", "precision", "recall", "f1"):
        logger.info("  %-10s %s", k, _fmt_mean_std(binary[k]))
    t = binary["totals"]
    logger.info(
        "  totals     TP=%d FP=%d TN=%d FN=%d  failed=%d",
        t["tp"],
        t["fp"],
        t["tn"],
        t["fn"],
        t["failed"],
    )

    logger.info("\n=== Multiclass (per-run-then-mean) ===")
    for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
        logger.info("  %-20s %s", k, _fmt_mean_std(multi[k]))
    logger.info("  failed_total         %d", multi["failed_total"])
    logger.info("  per-class (only GT-present classes):")
    for cls, vals in multi["per_class"].items():
        logger.info(
            "    %-32s P=%s  R=%s  F1=%s  support=%d",
            cls,
            _fmt_mean_std(vals["precision"]),
            _fmt_mean_std(vals["recall"]),
            _fmt_mean_std(vals["f1"]),
            vals["support"],
        )


def _fmt_mean_std(d: dict) -> str:
    if d.get("mean") is None:
        return "N/A"
    return f"{d['mean']:.3f}±{d['std']:.3f}"


def _fmt_cost_mean_std(d: dict) -> str:
    if d.get("mean") is None:
        return "N/A"
    return f"${d['mean']:.4f}±{d['std']:.4f}"


def _fmt_int_mean_std(d: dict) -> str:
    if d.get("mean") is None:
        return "N/A"
    return f"{d['mean']:,.0f}±{d['std']:,.0f}"


def register_parser(subparsers) -> None:
    p = subparsers.add_parser("metrics", help="Compute classification metrics from parsed.jsonl")
    add_experiment_args(p)
    p.set_defaults(func=lambda args: metrics(config_from_args(args)))
