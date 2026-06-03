import asyncio
import logging
import unittest

import aiohttp
import numpy as np

from sglang.srt.state_capturer.indexer_topk import (
    extract_indexer_topk_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=320, stage="extra-b", runner_config="8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# V3.2 config — hardcoded for response decoding (mirrors test_return_indexer_topk.py).
NUM_INDEXER_LAYERS = 61
INDEX_TOPK = 2048
INDEX_TOPK_FREQ = 2

logger = logging.getLogger(__name__)


class TestHiCacheReturnIndexerTopk(CustomTestCase):
    """HiCache + indexer-topk correctness e2e test for DSv3.2 (DSA) — issue #26975.

    Boots a server with BOTH `--enable-hierarchical-cache` and
    `--enable-return-indexer-topk`. This combination used to be rejected at
    startup; the INDEXER_TOPK sidecar host pool now migrates the captured topk
    alongside KV on host-tier cache hits.

    Flow:
      1. req_A: a long prompt -> collect its indexer-topk (the reference, produced
         by the original forward pass).
      2. A burst of distinct long filler prompts to evict req_A's KV from the
         device pool down to the host tier (write_backup).
      3. req_B: the SAME prompt as req_A -> its prefix is now served from the host
         cache (load_back -> INDEXER_TOPK sidecar restore into the capturer).
      4. Assert req_B's topk equals req_A's element-wise. A regression (sidecar not
         migrated) would surface as all -1 (unrestored) or mismatched (stale slots).

    Also keeps the shape / value-range / skip_topk(freq=2) invariants from
    test_return_indexer_topk.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--enable-return-indexer-topk",
            # The fix under test: HiCache must coexist with indexer-topk capture.
            "--enable-hierarchical-cache",
            # Small KV pool so filler prompts force host-tier eviction quickly, and
            # so the indexer-topk pinned buffer (~488 KB/token for V3.2) stays bounded.
            "--max-total-tokens",
            "32768",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
            "--json-model-override-args",
            f'{{"index_topk_freq": {INDEX_TOPK_FREQ}}}',
        ]
        cls.sampling_args = {"temperature": 0, "max_new_tokens": 8}

        # A long prompt that spans several KV pages so eviction + restore is
        # meaningfully exercised across pages.
        base = (
            "Describe in detail the history, geography, economy, and culture of a "
            "fictional country, covering its founding, major cities, rivers, "
            "mountains, trade routes, and notable historical figures. "
        )
        cls.shared_prompt = base * 12

        # Distinct long fillers to push the shared prompt's KV out of the device
        # pool and into the host tier between the two identical requests.
        cls.fillers = [f"Filler request number {i}. " + (base * 12) for i in range(24)]

        cls.process = popen_launch_server(
            DEEPSEEK_V32_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        try:
            cls.topk_a, cls.topk_b = asyncio.run(cls._collect_async())
        except Exception:
            kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @classmethod
    async def _collect_async(cls):
        async with aiohttp.ClientSession() as session:
            url = f"{DEFAULT_URL_FOR_TEST}/generate"

            # Phase 1: reference request (original forward pass).
            res_a = await _generate(session, url, cls.shared_prompt, cls.sampling_args)
            topk_a = _decode_topk(res_a)

            # Phase 2: evict req_A's KV to the host tier.
            await asyncio.gather(
                *[
                    _generate(session, url, f, cls.sampling_args, capture=False)
                    for f in cls.fillers
                ]
            )

            # Phase 3: identical request -> prefix served from host cache.
            res_b = await _generate(session, url, cls.shared_prompt, cls.sampling_args)
            topk_b = _decode_topk(res_b)
            return topk_a, topk_b

    def test_shape_and_range(self):
        for topk in (self.topk_a, self.topk_b):
            self.assertEqual(topk.ndim, 3)
            seqlen_minus_1, num_layers, topk_size = topk.shape
            self.assertGreater(seqlen_minus_1, 0)
            self.assertEqual(num_layers, NUM_INDEXER_LAYERS)
            self.assertEqual(topk_size, INDEX_TOPK)
            self.assertTrue((topk >= -1).all(), f"min index {topk.min()} < -1")

    def test_skip_topk_equality(self):
        for topk in (self.topk_a, self.topk_b):
            for L in range(2, NUM_INDEXER_LAYERS, 2):
                np.testing.assert_array_equal(
                    topk[:, L, :],
                    topk[:, L - 1, :],
                    err_msg=f"layer {L} should reuse layer {L - 1}'s topk",
                )

    def test_hicache_restore_matches_reference(self):
        """The core #26975 regression check.

        req_B and req_A are the identical prompt at temperature=0, so their
        indexer-topk must be byte-identical. req_B's prefix is reconstructed from
        the INDEXER_TOPK sidecar after a host-tier cache hit; any mismatch (or a
        prefix full of the -1 padding sentinel) means the captured topk was not
        migrated on load_back.
        """
        self.assertEqual(
            self.topk_a.shape,
            self.topk_b.shape,
            "identical prompts must yield identical topk shapes",
        )
        # Guard against the trivial all-padding failure mode on the restored side.
        self.assertTrue(
            (self.topk_b != -1).any(),
            "req_B topk is entirely the -1 padding sentinel: sidecar not restored",
        )
        np.testing.assert_array_equal(
            self.topk_b,
            self.topk_a,
            err_msg=(
                "HiCache-restored indexer-topk differs from the original forward "
                "pass — captured topk was not migrated on host-tier cache hit "
                "(issue #26975)."
            ),
        )


def _decode_topk(res) -> np.ndarray:
    return extract_indexer_topk_from_meta_info(res).reshape(
        -1, NUM_INDEXER_LAYERS, INDEX_TOPK
    )


async def _generate(session, url, text, sampling_params, capture: bool = True):
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "return_indexer_topk": capture,
    }
    async with session.post(url=url, json=payload) as response:
        return await response.json()


if __name__ == "__main__":
    unittest.main()
