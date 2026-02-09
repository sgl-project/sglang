"""MMMU evaluation tests for EPD (Encode-Prefill-Decode) disaggregated routing.

EPD disaggregation splits encoder, prefill, and decode across different
workers for improved throughput and resource utilization.

Requirements:
    - sgl_kernel package
    - GPUs: num_encode + num_prefill + num_decode (default: 8 GPUs for 6+1+1)
    - Optional: InfiniBand for high-performance transfers

Configuration via markers:
    @pytest.mark.model("model-id")  # Override default model
    @pytest.mark.workers(encode=6, prefill=1, decode=1)  # Custom worker counts
    @pytest.mark.gateway(policy="round_robin")  # Gateway configuration

Usage:
    # Basic (6 encode + 1 prefill + 1 decode)
    pytest e2e_test/router/test_epd_mmmu.py -v

    # Run specific test
    pytest e2e_test/router/test_epd_mmmu.py::TestEPDMMMU::test_epd_mmmu -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.workers(encode=6, prefill=1, decode=1)
@pytest.mark.parametrize("setup_backend", ["epd"], indirect=True)
class TestEPDMMMU:
    """MMMU evaluation tests using EPD disaggregated routing."""

    def test_epd_mmmu(self, setup_backend):
        """Basic MMMU evaluation with EPD disaggregation.

        Runs MMMU with 6 encode + 1 prefill + 1 decode worker and validates
        accuracy meets threshold (>= 0.45).
        """
        _, model, _, gateway = setup_backend

        args = SimpleNamespace(
            base_url=gateway.base_url,
            model=model,
            eval_name="mmmu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert (
            metrics["score"] >= 0.45
        ), f"EPD MMMU score {metrics['score']:.2f} below threshold 0.45"
        logger.info("EPD MMMU score: %.2f (threshold: 0.45)", metrics["score"])
