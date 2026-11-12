from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from home_made_rag.config import CONFIG


logger = logging.getLogger(__name__)


# Approximate public OpenAI prices in USD per 1M tokens. They can be overridden
# from config.json under "llm_pricing_usd_per_million_tokens".
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
}


def _ensure_console_logging() -> None:
    """Make sure cost logs reach the console when no logging setup exists."""
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


_ensure_console_logging()


def get_pricing(model_name: str) -> dict[str, float]:
    configured = (CONFIG.get("llm_pricing_usd_per_million_tokens") or {}).get(model_name)
    pricing = configured or DEFAULT_PRICING.get(model_name)

    if pricing is None:
        logger.warning(
            "[LLM] No pricing configured for model=%s; cost will be reported as 0",
            model_name,
        )
        return {"input": 0.0, "output": 0.0, "cached_input": 0.0}

    return {
        "input": float(pricing.get("input", 0.0)),
        "output": float(pricing.get("output", 0.0)),
        "cached_input": float(pricing.get("cached_input", pricing.get("input", 0.0))),
    }


def log_llm_call(
    model_name: str,
    call_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Log token usage and estimated cost for one LLM API call."""
    pricing = get_pricing(model_name)
    cached_tokens = max(0, min(int(cached_tokens), int(input_tokens)))
    uncached_tokens = max(0, int(input_tokens) - cached_tokens)

    input_cost = (
        uncached_tokens * pricing["input"]
        + cached_tokens * pricing["cached_input"]
    ) / 1_000_000
    output_cost = int(output_tokens) * pricing["output"] / 1_000_000
    total_cost = input_cost + output_cost

    total_tokens = int(input_tokens) + int(output_tokens)

    logger.info(
        "[LLM] call=%s model=%s input_tokens=%d cached_tokens=%d "
        "output_tokens=%d total_tokens=%d input_cost_usd=%.6f "
        "output_cost_usd=%.6f estimated_cost_usd=%.6f",
        call_name,
        model_name,
        int(input_tokens),
        cached_tokens,
        int(output_tokens),
        total_tokens,
        input_cost,
        output_cost,
        total_cost,
    )
    return total_cost


def _usage_from_token_usage(token_usage: dict[str, Any]) -> dict[str, int]:
    details = token_usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": int(
            token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        ),
        "output_tokens": int(
            token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        ),
        "cached_tokens": int(
            details.get("cached_tokens")
            or details.get("cache_read")
            or 0
        ),
    }


def _usage_from_usage_metadata(usage: dict[str, Any]) -> dict[str, int]:
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_tokens": int(
            details.get("cached_tokens")
            or details.get("cache_read")
            or 0
        ),
    }


def usage_from_message(message: Any) -> dict[str, int]:
    """Extract token usage from a LangChain AIMessage-like object."""
    if message is None:
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        return _usage_from_usage_metadata(usage_metadata)

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if token_usage:
        return _usage_from_token_usage(token_usage)

    return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


class LLMUsageLogger(BaseCallbackHandler):
    """LangChain callback that logs token usage and cost for each LLM call."""

    def __init__(self, model_name: str, call_name: str) -> None:
        self.model_name = model_name
        self.call_name = call_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        log_llm_call(
            self.model_name,
            self.call_name,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cached_tokens"],
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error(
            "[LLM] call=%s model=%s failed: %s estimated_cost_usd=0.000000",
            self.call_name,
            self.model_name,
            error,
        )

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if token_usage:
            return _usage_from_token_usage(token_usage)

        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                usage = usage_from_message(getattr(generation, "message", None))
                if usage["input_tokens"] or usage["output_tokens"]:
                    return usage

        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
