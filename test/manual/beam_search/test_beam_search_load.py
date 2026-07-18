"""Beam search load and admission-saturation tests.

Workload specs from the beam search PR verification (#27716), on
Qwen2.5-0.5B (override with SGLANG_TEST_BEAM_LOAD_MODEL) with
--enable-beam-search --disable-radix-cache:

- Mixed-width load: 100 requests at 10 QPS, beam widths drawn from [2, 100];
  expect 100/100 OK, report p50/p90/p99 latency (reference: 20/24/24 ms).
- Extreme fanout: n=3200 at 4 QPS for 60s; each request needs 3201
  req-to-token slots so the admission gate must serialize them; expect
  240/240 OK with a stable server (reference p50/p90/p99: 290/525/752 ms).

Manual test (not registered in CI). Run on a GPU host:
    python3 test_beam_search_load.py
"""

import asyncio
import os
import random
import time
import unittest

import aiohttp
import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MAX_NEW_TOKENS = 10
CLIENT_TIMEOUT_S = 600
PROMPT = "Write a short story about a robot learning to paint."


async def _generate(session, base_url, width):
    start = time.perf_counter()
    async with session.post(
        f"{base_url}/generate",
        json={
            "text": PROMPT,
            "sampling_params": {"n": width, "max_new_tokens": MAX_NEW_TOKENS},
        },
    ) as resp:
        payload = await resp.json()
    latency = time.perf_counter() - start
    beam_results = payload.get("meta_info", {}).get("beam_results") or []
    return resp.status, len(beam_results), latency


async def _run_at_qps(base_url, widths, qps):
    """Fire one request per width at a fixed rate; return per-request results."""
    timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def delayed(i, width):
            await asyncio.sleep(i / qps)
            return await _generate(session, base_url, width)

        return await asyncio.gather(
            *[delayed(i, width) for i, width in enumerate(widths)]
        )


def _report_latencies(name, results):
    latencies_ms = [lat * 1000 for _, _, lat in results]
    p50, p90, p99 = np.percentile(latencies_ms, [50, 90, 99])
    print(f"{name}: n={len(results)}  p50/p90/p99 = {p50:.0f}/{p90:.0f}/{p99:.0f} ms")


class _BeamLoadTestBase(CustomTestCase):
    extra_server_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_BEAM_LOAD_MODEL", "Qwen/Qwen2.5-0.5B")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-beam-search", "--disable-radix-cache"]
            + cls.extra_server_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _check_all_ok(self, results, widths):
        num_ok = sum(1 for status, _, _ in results if status == 200)
        self.assertEqual(num_ok, len(results), f"{len(results) - num_ok} failed")
        for (_, num_beams, _), width in zip(results, widths):
            self.assertGreaterEqual(num_beams, 1)
            self.assertLessEqual(num_beams, width)


class TestBeamSearchMixedWidthLoad(_BeamLoadTestBase):
    """100 requests at 10 QPS with beam widths mixed in [2, 100]."""

    def test_mixed_width_load(self):
        rng = random.Random(42)
        widths = [rng.randint(2, 100) for _ in range(100)]
        results = asyncio.run(_run_at_qps(self.base_url, widths, qps=10))
        self._check_all_ok(results, widths)
        _report_latencies("mixed-width 100 reqs @ 10 QPS", results)


class TestBeamSearchExtremeFanout(_BeamLoadTestBase):
    """n=3200 at 4 QPS for 60s: admission gate must serialize, not crash."""

    # Each n=3200 request owns 3201 req-to-token slots; make the pool size
    # deterministic so exactly one group fits at a time.
    extra_server_args = ["--max-running-requests", "4000"]

    def test_extreme_fanout(self):
        duration_s = 60
        qps = 4
        widths = [3200] * (duration_s * qps)
        results = asyncio.run(_run_at_qps(self.base_url, widths, qps=qps))
        self._check_all_ok(results, widths)
        _report_latencies(f"extreme fanout n=3200 @ {qps} QPS", results)

        # Server must still be alive and serving after the burst.
        final = asyncio.run(_run_at_qps(self.base_url, [2], qps=1))
        self._check_all_ok(final, [2])


if __name__ == "__main__":
    unittest.main()
