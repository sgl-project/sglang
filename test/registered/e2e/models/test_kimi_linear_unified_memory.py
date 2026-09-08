"""Kimi-Linear (MLA full attention + KDA linear attention) served from the
unified memory pool.

Under `--enable-unified-memory` the MLA full side is exposed as per-layer views
and every loc the kernels see is a translated virtual id, so the whole
read/write path differs from the static pool. `test_prefix_cache_branching`
carries most of the weight: a radix hit replays virtual locs whose physical
pages may have moved under compaction.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=800, stage="nightly", runner_config="4-gpu-h100")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinearUnifiedMemoryFlashMLA(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    """flashmla at its ps=64 snap: the block-table route
    (KVIndexTranslator.fill_read_table into flashmla's padded tables) plus the
    ps=64 sub-pool sizing (64-token sink floor, per-layer-view tail pad).
    Hopper-only, like the rest of this nightly suite."""

    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    # Same bar as the static-pool Kimi-Linear e2e test: unified memory must not
    # cost accuracy.
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--enable-unified-memory",
        "--attention-backend",
        "flashmla",
        "--page-size",
        "64",
    ]


class TestKimiLinearUnifiedMemoryDCP(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    """Unified memory + decode context parallelism on flashinfer: on a radix hit
    each rank must recover the same physical page from widened virtual locs
    while keeping a different row inside it."""

    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--dcp-size",
        "2",
        "--attention-backend",
        "flashinfer",
        "--chunked-prefill-size",
        "2048",
        "--enable-unified-memory",
    ]


if __name__ == "__main__":
    unittest.main()
