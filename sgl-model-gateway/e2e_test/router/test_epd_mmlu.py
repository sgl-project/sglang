"""EPD routing tests for text-only paths (PD fallback)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import requests
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.model("qwen-vl-7b")
@pytest.mark.workers(encode=1, prefill=1, decode=1)
@pytest.mark.parametrize("setup_backend", ["epd"], indirect=True)
class TestEPDMMLU:
    """EPD text-only tests using PD fallback routing."""

    def test_epd_simple_text(self, setup_backend):
        """Simple chat completion for EPD pipeline sanity."""
        _, model, _, gateway = setup_backend

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer with just the number.",
                    }
                ],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=60,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert "4" in content, f"Expected '4' in response, got: {content}"
        logger.info("Simple text test passed. Response: %s", content)

    def test_epd_mmlu_basic(self, setup_backend):
        """Run MMLU (text-only fallback) with EPD routing."""
        _, model, _, gateway = setup_backend

        args = SimpleNamespace(
            base_url=gateway.base_url,
            model=model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert metrics["score"] >= 0.50, (
            f"EPD MMLU score {metrics['score']:.2f} below threshold 0.50"
        )
