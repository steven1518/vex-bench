"""Per-model token pricing — turns a run's ``RunStats`` into a USD cost.

A benchmark run's only cross-agent-comparable efficiency number is its
dollar cost. Raw token counts are not comparable: a cached token, a fresh
prompt token and an output token bill at different rates, and rates differ
by an order of magnitude across providers.

``RunStats`` already normalises every agent onto five disjoint token
buckets; ``cost_usd`` multiplies those buckets by the per-model rates
below. Reasoning tokens are billed at the model's output rate (the
near-universal provider convention), so each entry carries four rates,
not five.

Rates are USD per 1M tokens. An unpriced model yields ``cost_usd = None``
rather than a misleading 0.0, so partial pricing coverage is visible
instead of silently wrong.
"""

import logging
from dataclasses import dataclass

from evaluate.agents.base import RunStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens for each disjoint billing bucket."""

    input: float  # fresh prompt tokens (not cache hits)
    cached_input: float  # cache-read prompt tokens
    cache_write: float  # cache-creation prompt tokens
    output: float  # completion tokens; reasoning tokens bill at this rate too


# ExperimentConfig.model -> rates, USD per 1M tokens. Keep in sync with the
# provider price sheets. Rates below are list/official prices as of 2026-05;
# runs routed through a proxy (the opencode Azure-Resource provider) may
# bill differently -- verify against the actual invoice.
#
# cache_write is set to the input rate: DeepSeek and Moonshot bill
# cache-creation tokens at the normal cache-miss rate (no Anthropic-style
# premium), and Codex never reports cache writes at all.
PRICING: dict[str, ModelPrice] = {
    "gpt-5.5": ModelPrice(input=5.00, cached_input=0.50, cache_write=5.00, output=30.00),
    "gpt-5.4-mini": ModelPrice(input=0.75, cached_input=0.075, cache_write=0.75, output=4.50),
    "kimi-k2.6": ModelPrice(input=0.60, cached_input=0.10, cache_write=0.60, output=2.50),
    "deepseek-v4-pro": ModelPrice(input=0.435, cached_input=0.003625, cache_write=0.435, output=0.87),
    "deepseek-v4-flash": ModelPrice(input=0.14, cached_input=0.0028, cache_write=0.14, output=0.28),
    "claude-opus-4-6": ModelPrice(input=5.00, cached_input=0.50, cache_write=6.25, output=25.00),
    "claude-sonnet-4-6": ModelPrice(input=3.00, cached_input=0.30, cache_write=3.75, output=15.00),
    "claude-haiku-4-5": ModelPrice(input=1.00, cached_input=0.10, cache_write=1.25, output=5.00),
}


def cost_usd(stats: RunStats, model: str) -> float | None:
    """USD cost of one run, or None if it cannot be priced.

    Returns None when the model has no ``PRICING`` entry, or when no token
    counts were parsed at all — callers render N/A instead of a wrong 0.0.
    """
    price = PRICING.get(model)
    if price is None:
        logger.debug("no pricing for model %r; cost_usd left None", model)
        return None
    if all(
        b is None
        for b in (
            stats.input_tokens,
            stats.cached_input_tokens,
            stats.cache_write_tokens,
            stats.output_tokens,
            stats.reasoning_tokens,
        )
    ):
        return None
    return (
        (stats.input_tokens or 0) * price.input
        + (stats.cached_input_tokens or 0) * price.cached_input
        + (stats.cache_write_tokens or 0) * price.cache_write
        + ((stats.output_tokens or 0) + (stats.reasoning_tokens or 0)) * price.output
    ) / 1_000_000
