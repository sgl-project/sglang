"""
E2E test for MLA HiCache host dedup with rotating layer owners
(--enable-mla-hicache-host-dedup) at TP2.

A small device pool makes GSM8K prompts evict to the host tier, where each rank
keeps only its own layers; the second pass loads them back and every layer is
broadcast from its owner. Accuracy must hold and the load-back must happen.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_mla_host_dedup.py -v
"""

import re
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


def _metric_total(metrics_text: str, name: str) -> float:
    """Largest per-rank sample of a counter: each rank reports the same logical count."""
    values = [
        float(m.group(1))
        for m in re.finditer(
            rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
            metrics_text,
            re.MULTILINE,
        )
    ]
    return max(values, default=0.0)


class TestHiCacheMLAHostDedup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                2,
                "--enable-hierarchical-cache",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-write-policy",
                "write_back",
                "--hicache-ratio",
                4,
                # Far fewer device tokens than one GSM8K pass, so prompts are
                # evicted to host and loaded back on the second pass.
                "--max-total-tokens",
                16384,
                "--enable-mla-hicache-host-dedup",
                "--enable-metrics",
            ],
            env={"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def _load_back_tokens(self) -> float:
        response = requests.get(f"{self.base_url}/metrics", timeout=30)
        response.raise_for_status()
        return _metric_total(response.text, "sglang:load_back_tokens_total")

    def test_gsm8k_after_host_load_back(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=32,
        )
        first = run_eval(args)
        loaded_before = self._load_back_tokens()
        second = run_eval(args)
        loaded_after = self._load_back_tokens()
        print(
            f"GSM8K first={first['score']:.4f} second={second['score']:.4f} "
            f"load-back tokens during second pass={loaded_after - loaded_before:.0f}"
        )

        self.assertGreater(
            loaded_after - loaded_before,
            0,
            "The second pass should load evicted prefixes back from host",
        )
        self.assertGreater(first["score"], 0.6)
        self.assertGreater(second["score"], 0.6)
        self.assertLess(
            abs(first["score"] - second["score"]),
            0.05,
            "Accuracy should hold when KV is loaded back through the dedup broadcast",
        )


if __name__ == "__main__":
    unittest.main()
