"""EPD routing tests for multimodal MMMU evaluation."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.model("qwen-vl-7b")
@pytest.mark.workers(encode=1, prefill=1, decode=1)
@pytest.mark.parametrize("setup_backend", ["epd"], indirect=True)
class TestEPDMMMU:
    """MMMU evaluation using EPD routing."""

    def test_epd_mmmu(self, setup_backend):
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
