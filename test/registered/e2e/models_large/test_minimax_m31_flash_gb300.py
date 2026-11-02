"""MiniMax-M3.1-Flash on GB300 with the B300 balanced recipe, DSpark and HiCache.

Covers the Blackwell path: MSA sparse attention, the TRT-LLM NVFP4 MoE, the
bundled DSpark draft, and HiCache moving the NVFP4 page rows (packed KV, block
scales, index K and draft KV) between GPU, host and file storage.
"""

import re
import shutil
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=2200, stage="nightly", runner_config="4-gpu-gb300")

_STORAGE_HIT_METRIC = re.compile(
    r"^sglang:storage_prefetch_hit_tokens_total(?:\{[^}]*\})? (\S+)$", re.M
)


class TestMiniMaxM31FlashGB300(GSM8KMixin, DefaultServerBase):
    model = "MiniMaxAI/MiniMax-M3.1-Flash"
    other_args = [
        "--trust-remote-code",
        "--reasoning-parser",
        "minimax-m3",
        "--tool-call-parser",
        "minimax-m3",
        "--tp",
        "4",
        "--moe-runner-backend",
        "flashinfer_trtllm_routed",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--chunked-prefill-size",
        "32768",
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs-decode",
        "128",
        "--mem-fraction-static",
        "0.8",
        "--enable-hierarchical-cache",
        "--hicache-write-policy",
        "write_through",
        "--hicache-storage-backend",
        "file",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
        "--enable-metrics",
    ]

    gsm8k_backend = "sgl_eval"
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95
    # Measured 0.961 / 0.961 and accept length 4.22 / 4.18 on 4x B300.
    gsm8k_score_threshold = 0.93
    gsm8k_accept_length_thres = 3.8

    @classmethod
    def setUpClass(cls):
        cls.storage_dir = tempfile.mkdtemp(prefix="m31-hicache-")
        cls.server_env = {
            "FMHA_SM100_ALLOW_JIT": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
        }
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _storage_hit_tokens(self) -> float:
        text = requests.get(f"{self.base_url}/metrics", timeout=30).text
        return sum(float(v) for v in _STORAGE_HIT_METRIC.findall(text))

    def _generate(self, prompt: str) -> str:
        res = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=300,
        )
        res.raise_for_status()
        return res.json()["text"]

    def test_prefix_reloaded_from_storage_matches(self):
        # Well past the 256-token storage prefetch threshold and several pages long.
        prompt = (
            "".join(
                f"Ledger entry {i}: warehouse {i % 7} shipped {i * 13 % 101} crates "
                f"to port {i % 11} on day {i % 29}.\n"
                for i in range(400)
            )
            + "Question: how many crates did warehouse 3 ship on entry 101? Answer:"
        )
        self._generate(prompt)
        # A cold prefill and a cache hit take different numeric paths, so compare
        # against a GPU hit, which reads the NVFP4 pages the storage reload restores.
        gpu_hit = self._generate(prompt)
        requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=40
        ).raise_for_status()
        hits_before = self._storage_hit_tokens()
        reloaded = self._generate(prompt)

        self.assertGreater(self._storage_hit_tokens(), hits_before)
        self.assertEqual(gpu_hit, reloaded)


if __name__ == "__main__":
    unittest.main()
