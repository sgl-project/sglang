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
            "sampling_params": {"beam_width": width, "max_new_tokens": MAX_NEW_TOKENS},
        },
    ) as resp:
        payload = await resp.json()
    latency = time.perf_counter() - start
    beam_results = payload.get("meta_info", {}).get("beam_results") or []
    return resp.status, len(beam_results), latency


class _BeamSweepBase(CustomTestCase):
    """Shared sweep harness; concrete classes pick the pool configuration.

    Primary metric is aggregate beam tok/s (= reqs x width x new_tokens /
    elapsed); QPS is reported secondarily since it conflates width.
    """

    extra_server_args = []
    pool_label = "default pool"

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_BEAM_MODEL", "Qwen/Qwen3-1.7B")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # Leave VRAM headroom: the beam logprob path materializes a
            # full-vocab [num_beam_rows, vocab] logprobs tensor each step,
            # which OOMs at large width x concurrency if the KV pool takes the
            # default share.
            other_args=["--disable-overlap-schedule", "--mem-fraction-static", "0.7"]
            + cls.extra_server_args,
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

    def _run_sweep(self):
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

            beam_tok_s = NUM_PROMPTS * width * MAX_NEW_TOKENS / elapsed
            qps = NUM_PROMPTS / elapsed
            report.append((width, beam_tok_s, qps, elapsed))
            print(
                f"width={width:4d}  beam_tok/s={beam_tok_s:9.0f}  "
                f"qps={qps:6.2f}  elapsed={elapsed:6.2f}s"
            )

        print(f"\nBeam width sweep ({self.pool_label}):")
        print("| beam width | beam tok/s | qps | elapsed (s) |")
        print("|---|---|---|---|")
        for width, beam_tok_s, qps, elapsed in report:
            print(f"| {width} | {beam_tok_s:.0f} | {qps:.2f} | {elapsed:.2f} |")


class TestBeamSweepDefaultPool(_BeamSweepBase):
    """Default req-slot pool: measures the deployment-default curve.

    With the default max_running_requests (4096) the in-flight beam rows pin
    at ~4000 for width >= 50, so this curve saturates at the pool, not the
    engine.
    """

    def test_beam_width_sweep(self):
        self._run_sweep()


class TestBeamSweepLargePool(_BeamSweepBase):
    """Enlarged req-slot pool: measures the engine ceiling.

    16384 slots let up to ~40 width-400 groups run concurrently; the short
    --context-length keeps the req_to_token pool small enough to afford it.
    """

    extra_server_args = [
        "--max-running-requests",
        "16384",
        "--context-length",
        "2048",
    ]
    pool_label = "large pool (16384 slots)"

    def test_beam_width_sweep(self):
        self._run_sweep()


if __name__ == "__main__":
    unittest.main()
