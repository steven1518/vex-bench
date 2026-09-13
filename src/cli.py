"""Top-level CLI dispatcher.

Builds a two-level argparse subcommand tree:

    vex-bench evaluate {run, parse, metrics}
    vex-bench download

Each leaf module under ``src/`` exposes a ``register_parser(subparsers)``
function that attaches its parser and binds its callback via
``set_defaults(func=...)``. This module wires those together.
"""

import argparse
import logging

from builder import download as download_cmd
from evaluate import metrics as evaluate_metrics
from evaluate import parse as evaluate_parse
from evaluate import run as evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="vex-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    evaluate = sub.add_parser("evaluate", help="Run / parse / score the evaluation pipeline")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_cmd", required=True)
    evaluate_run.register_parser(evaluate_sub)
    evaluate_parse.register_parser(evaluate_sub)
    evaluate_metrics.register_parser(evaluate_sub)

    download_cmd.register_parser(sub)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.func(args)
