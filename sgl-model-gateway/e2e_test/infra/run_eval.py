"""MMLU evaluation runner for E2E tests.

Simplified evaluation runner that uses local eval implementations
with cleaner logging for CI/CD environments.

Usage:
    from infra.run_eval import run_eval
    from types import SimpleNamespace

    args = SimpleNamespace(
        base_url="http://127.0.0.1:30000",
        model="meta-llama/Llama-3.1-8B-Instruct",
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simple_eval_common import Eval

from .simple_eval_common import ChatCompletionSampler, set_ulimit

logger = logging.getLogger(__name__)

# MMLU dataset URL
MMLU_DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"


@dataclass
class EvalConfig:
    """Configuration for running an evaluation."""

    base_url: str
    model: str | None = None
    eval_name: str = "mmlu"
    num_examples: int = 64
    num_threads: int = 32
    temperature: float = 0.0
    max_tokens: int = 2048
    host: str = "127.0.0.1"
    port: int = 30000


def _get_eval(eval_name: str, num_examples: int, num_threads: int) -> "Eval":
    """Get the evaluation object by name."""
    if eval_name == "mmlu":
        from .simple_eval_mmlu import MMLUEval

        return MMLUEval(MMLU_DATASET_URL, num_examples, num_threads)
    else:
        raise ValueError(f"Unknown eval: {eval_name}. Supported: mmlu")


def run_eval(args: Any) -> dict:
    """Run an evaluation and return metrics.

    Args:
        args: Configuration object with attributes:
            - base_url: Base URL of the server (e.g., "http://127.0.0.1:30000")
            - model: Model name/path (optional, will be auto-detected)
            - eval_name: Evaluation name ("mmlu")
            - num_examples: Number of examples to evaluate
            - num_threads: Number of parallel threads
            - temperature: Sampling temperature

    Returns:
        Dict with metrics including 'score' key.
    """
    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    # Build base URL
    base_url = getattr(args, "base_url", None)
    if base_url:
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
    else:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 30000)
        base_url = f"http://{host}:{port}/v1"

    eval_name = getattr(args, "eval_name", "mmlu")
    num_examples = getattr(args, "num_examples", 64)
    num_threads = getattr(args, "num_threads", 32)
    temperature = getattr(args, "temperature", 0.0)
    max_tokens = getattr(args, "max_tokens", 2048)
    model = getattr(args, "model", None)

    logger.info(
        "Starting %s eval: %d examples, %d threads, temp=%.2f",
        eval_name,
        num_examples,
        num_threads,
        temperature,
    )

    # Create sampler
    sampler = ChatCompletionSampler(
        model=model,
        max_tokens=max_tokens,
        base_url=base_url,
        temperature=temperature,
    )

    # Get eval object
    eval_obj = _get_eval(eval_name, num_examples, num_threads)

    # Run evaluation
    start_time = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - start_time

    # Build metrics
    metrics = result.metrics.copy() if result.metrics else {}
    metrics["score"] = result.score
    metrics["latency"] = latency

    logger.info(
        "%s eval complete: score=%.3f, latency=%.1fs, model=%s",
        eval_name,
        result.score,
        latency,
        sampler.model,
    )

    return metrics
