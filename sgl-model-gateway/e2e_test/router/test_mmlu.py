"""MMLU evaluation tests for router functionality.

Tests the router's ability to handle MMLU benchmark evaluations across
different backend configurations (gRPC and HTTP workers).

Usage:
    # Run with gRPC backend only
    pytest e2e_test/router/test_mmlu.py -v

    # Run with specific backend
    pytest e2e_test/router/test_mmlu.py -v -k "grpc"
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestMMLU:
    """MMLU evaluation tests using local workers (gRPC and HTTP)."""

    def test_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with score threshold.

        Runs MMLU evaluation with 64 examples and validates that
        accuracy meets minimum threshold (>= 0.65).

        Note: setup_backend fixture already waits for workers to be ready.
        """
        backend, model, client = setup_backend
        base_url = str(client.base_url).rstrip("/v1")

        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert (
            metrics["score"] >= 0.65
        ), f"MMLU score {metrics['score']:.2f} below threshold 0.65"
        logger.info("MMLU score: %.2f (threshold: 0.65)", metrics["score"])

    def test_mmlu_extended(self, setup_backend):
        """Extended MMLU evaluation with more examples.

        Runs MMLU with 128 examples for more statistically
        significant results.
        """
        backend, model, client = setup_backend
        base_url = str(client.base_url).rstrip("/v1")

        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=64,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert (
            metrics["score"] >= 0.65
        ), f"MMLU score {metrics['score']:.2f} below threshold 0.65"
        logger.info("MMLU extended score: %.2f (threshold: 0.65)", metrics["score"])
