"""Beam width QPS sweep benchmark.

Workload spec from the beam search PR verification (#15645):
- Model: Qwen3-1.7B (override with SGLANG_TEST_BEAM_MODEL)
- Dataset: ShareGPT, 100 samples filtered to prompt_len < 100 tokens
- max_new_tokens=10, beam widths swept over 10 / 50 / 100 / 200 / 400
- Metric: per-width QPS (100 concurrent requests / wall time)

Manual test (not registered in CI). Run on a GPU host:
    python3 test_beam_search_perf_sweep.py
"""

import asyncio
import os
import time
import unittest

import aiohttp

from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

BEAM_WIDTHS = [10, 50, 100, 200, 400]
NUM_PROMPTS = 100
MAX_PROMPT_LEN = 100
MAX_NEW_TOKENS = 10
CLIENT_TIMEOUT_S = 1200


async def _generate(session, base_url, prompt, width):
    start = time.perf_counter()
    async with session.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"n": width, "max_new_tokens": MAX_NEW_TOKENS},
        },
    ) as resp:
        payload = await resp.json()
    latency = time.perf_counter() - start
    beam_results = payload.get("meta_info", {}).get("beam_results") or []
    return resp.status, len(beam_results), latency


class TestBeamSearchPerfSweep(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_BEAM_MODEL", "Qwen/Qwen3-1.7B")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-beam-search"],
        )

        tokenizer = get_tokenizer(cls.model)
        rows = sample_sharegpt_requests(
            dataset_path="", num_requests=4000, tokenizer=tokenizer
        )
        cls.prompts = [r.prompt for r in rows if r.prompt_len < MAX_PROMPT_LEN][
            :NUM_PROMPTS
        ]
        assert (
            len(cls.prompts) == NUM_PROMPTS
        ), f"only {len(cls.prompts)} short prompts sampled"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    async def _run_one_width(self, width):
        timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.perf_counter()
            results = await asyncio.gather(
                *[
                    _generate(session, self.base_url, prompt, width)
                    for prompt in self.prompts
                ]
            )
            elapsed = time.perf_counter() - start
        return results, elapsed

    def test_beam_width_qps_sweep(self):
        report = []
        for width in BEAM_WIDTHS:
            results, elapsed = asyncio.run(self._run_one_width(width))

            num_ok = sum(1 for status, _, _ in results if status == 200)
            self.assertEqual(
                num_ok, NUM_PROMPTS, f"width={width}: {NUM_PROMPTS - num_ok} failed"
            )
            for status, num_beams, _ in results:
                self.assertGreaterEqual(num_beams, 1)
                self.assertLessEqual(num_beams, width)

            qps = NUM_PROMPTS / elapsed
            report.append((width, qps, elapsed))
            print(f"width={width:4d}  qps={qps:6.2f}  elapsed={elapsed:6.2f}s")

        print("\nBeam width QPS sweep:")
        print("| beam width | qps | elapsed (s) |")
        print("|---|---|---|")
        for width, qps, elapsed in report:
            print(f"| {width} | {qps:.2f} | {elapsed:.2f} |")


if __name__ == "__main__":
    unittest.main()
