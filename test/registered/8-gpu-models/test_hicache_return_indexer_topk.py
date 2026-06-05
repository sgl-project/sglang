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

register_cuda_ci(est_time=420, stage="extra-b", runner_config="8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# V3.2 config — hardcoded for response decoding (mirrors test_return_indexer_topk.py).
NUM_INDEXER_LAYERS = 61
INDEX_TOPK = 2048
INDEX_TOPK_FREQ = 2

# Topology / eviction tuning. IndexerTopkCapturer requires attn_tp_size == 1, so
# dp_size must equal tp_size (=> dp 8 on 8 GPUs). KV pools are PER-RANK, and the
# default load balancer is round_robin, so a request and its identical re-send do
# NOT land on the same rank. We therefore:
#   - keep the per-rank device pool tiny so a modest filler burst evicts req_A to
#     the host tier on whichever rank served it, and
#   - probe req_B up to DP_SIZE times: round_robin visits each rank exactly once
#     per DP_SIZE consecutive sends, so the single probe that lands on req_A's
#     rank is the one that triggers load_back -> INDEXER_TOPK sidecar restore.
# A host-tier hit (cached_tokens_details["host"] > 0) is ONLY possible on req_A's
# rank, since no other rank ever held the shared prompt's KV — so host > 0 both
# proves the restore path ran and pins us to the correct rank.
DP_SIZE = 8
MAX_TOTAL_TOKENS = 4096  # per-rank device KV pool; small enough to force eviction.
CHUNKED_PREFILL_SIZE = 2048  # must be <= MAX_TOTAL_TOKENS on a tiny pool.
NUM_FILLERS = 64  # ~8 per rank, each ~MAX_TOTAL_TOKENS-worth, evicts req_A.
SETTLE_SECONDS = 5  # let the async HiCache write_backup flush before probing.
MAX_PROBES = DP_SIZE  # round_robin guarantees req_A's rank is visited within DP_SIZE.

logger = logging.getLogger(__name__)


class TestHiCacheReturnIndexerTopk(CustomTestCase):
    """HiCache + indexer-topk correctness e2e test for DSv3.2 (DSA) — issue #26975.

    Boots a server with BOTH `--enable-hierarchical-cache` and
    `--enable-return-indexer-topk`. This combination used to be rejected at
    startup; the INDEXER_TOPK sidecar host pool now migrates the captured topk
    alongside KV on host-tier cache hits.

    Flow:
      1. req_A: a long prompt -> collect its indexer-topk (the reference, produced
         by the original forward pass) and the rank that served it.
      2. A burst of distinct long filler prompts to evict req_A's KV from every
         rank's device pool down to the host tier (write_backup).
      3. req_B: the SAME prompt as req_A, re-sent until a probe reports a
         host-tier cache hit (load_back -> INDEXER_TOPK sidecar restore). Only
         req_A's rank can produce a host hit for this prompt.
      4. Assert req_B's restored prefix equals req_A's element-wise. A regression
         (sidecar not migrated) surfaces as all -1 (unrestored) or mismatched
         (stale slots) on the restored rows — and is no longer maskable by a
         silent recompute, because we require an actual host hit.

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
            # Pin the balancer so the probe-until-host-hit logic below holds.
            "--load-balance-method",
            "round_robin",
            # Tiny per-rank KV pool so the filler burst forces host-tier eviction
            # quickly, and so the indexer-topk pinned buffer stays bounded.
            "--max-total-tokens",
            str(MAX_TOTAL_TOKENS),
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
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

        # Distinct long fillers (no shared prefix with shared_prompt) to push the
        # shared prompt's KV out of every rank's device pool into the host tier.
        cls.fillers = [
            f"Filler request number {i}. " + (base * 12) for i in range(NUM_FILLERS)
        ]

        cls.process = popen_launch_server(
            DEEPSEEK_V32_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        try:
            (
                cls.topk_a,
                cls.topk_b,
                cls.cached_len_b,
                cls.rank_a,
                cls.rank_b,
                cls.probes,
            ) = asyncio.run(cls._collect_async())
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

            # Phase 1: reference request (original forward pass, must be a cold
            # miss so the captured topk is genuinely produced by forward).
            res_a = await _generate(session, url, cls.shared_prompt, cls.sampling_args)
            if _cached_tokens(res_a) != 0:
                raise AssertionError(
                    "reference req_A was a cache hit "
                    f"(cached_tokens={_cached_tokens(res_a)}); expected a cold miss"
                )
            topk_a = _decode_topk(res_a)
            rank_a = _dp_rank(res_a)

            # Phase 2: evict req_A's KV to the host tier on every rank.
            await asyncio.gather(
                *[
                    _generate(session, url, f, cls.sampling_args, capture=False)
                    for f in cls.fillers
                ]
            )
            # Let the async HiCache write_backup flush before we probe.
            await asyncio.sleep(SETTLE_SECONDS)

            # Phase 3: re-send the identical prompt until a probe reports a
            # host-tier hit. Only req_A's rank holds this prompt's backed-up KV,
            # so host > 0 both confirms the restore ran and pins us to that rank.
            res_b = None
            probes = []
            for _ in range(MAX_PROBES):
                r = await _generate(session, url, cls.shared_prompt, cls.sampling_args)
                probes.append((_dp_rank(r), _cached_tokens(r), _host_cached(r)))
                if _host_cached(r) > 0:
                    res_b = r
                    break

            if res_b is None:
                raise AssertionError(
                    "no host-tier cache hit observed across "
                    f"{MAX_PROBES} probes (rank_a={rank_a}; "
                    f"probes[(rank, cached, host)]={probes}). req_A was likely not "
                    "evicted to the host tier — increase NUM_FILLERS / SETTLE_SECONDS "
                    "or lower MAX_TOTAL_TOKENS."
                )

            topk_b = _decode_topk(res_b)
            return (
                topk_a,
                topk_b,
                _cached_tokens(res_b),
                rank_a,
                _dp_rank(res_b),
                probes,
            )

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

        req_B's prefix is reconstructed from the INDEXER_TOPK sidecar after a
        host-tier cache hit (guaranteed: setUpClass only accepts a probe with
        host > 0). The restored prefix rows must byte-equal req_A's reference at
        the same positions. We compare ONLY the cache-served prefix region: the
        recomputed tail goes through the extend path rather than full prefill and
        can differ in the last index of a near-tie, so comparing it would be
        flaky and is not what #26975 is about.

        A mismatch (or a prefix of the -1 padding sentinel) means the captured
        topk was not migrated on load_back.
        """
        # Host-tier hit actually happened (this is what makes the test able to
        # detect the regression at all — a silent recompute would pass trivially).
        host = self.probes[-1][2]
        self.assertGreater(
            host, 0, f"req_B did not hit the host tier; probes={self.probes}"
        )
        if self.rank_a is not None and self.rank_b is not None:
            self.assertEqual(
                self.rank_b,
                self.rank_a,
                "host hit must be on req_A's rank (only it holds the prompt's KV)",
            )

        # Compare the cache-served prefix region row-for-row.
        n = min(self.cached_len_b, self.topk_a.shape[0], self.topk_b.shape[0])
        self.assertGreater(n, 0, "no cached prefix rows to compare on req_B")
        prefix_a = self.topk_a[:n]
        prefix_b = self.topk_b[:n]

        # Guard against the trivial all-padding failure mode on the restored side.
        self.assertTrue(
            (prefix_b != -1).any(),
            "req_B restored prefix is entirely the -1 padding sentinel: "
            "sidecar not restored",
        )
        np.testing.assert_array_equal(
            prefix_b,
            prefix_a,
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


def _cached_tokens(res) -> int:
    return int(res["meta_info"].get("cached_tokens", 0))


def _host_cached(res) -> int:
    details = res["meta_info"].get("cached_tokens_details") or {}
    return int(details.get("host", 0))


def _dp_rank(res):
    return res["meta_info"].get("dp_rank")


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
