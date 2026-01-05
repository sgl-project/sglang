"""MMLU evaluation tests for PD (Prefill-Decode) disaggregated routing.

PD disaggregation separates prefill and decode phases across different
workers for improved throughput and resource utilization.

Requirements:
    - sgl_kernel package
    - GPUs: num_prefill + num_decode (default: 2 GPUs for 1+1)
    - Optional: InfiniBand for high-performance transfers

Configuration via markers:
    @pytest.mark.model("model-id")  # Override default model
    @pytest.mark.pd(num_prefill=2, num_decode=2)  # Custom worker counts

Usage:
    # Basic (1 prefill + 1 decode)
    pytest e2e_test/router/test_pd_mmlu.py -v

    # Run specific test
    pytest e2e_test/router/test_pd_mmlu.py::TestPDMMLU::test_pd_mmlu_basic -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
class TestPDMMLU:
    """MMLU evaluation tests using PD disaggregated routing."""

    def test_pd_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with PD disaggregation.

        Runs MMLU with 1 prefill + 1 decode worker and validates
        accuracy meets threshold (>= 0.65).
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
        ), f"PD MMLU score {metrics['score']:.2f} below threshold 0.65"
        logger.info("PD MMLU score: %.2f (threshold: 0.65)", metrics["score"])
