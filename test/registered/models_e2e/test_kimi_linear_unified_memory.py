"""Kimi-Linear (MLA full attention + KDA linear attention) served from the
unified memory pool.

`--enable-unified-memory` replaces the statically-partitioned hybrid pools with
one byte buffer split dynamically between the full-attention KV sub-pool and the
per-request KDA state sub-pool. For an MLA model the full side is exposed as
DENSE per-layer views (`build_dense_mla_views`) and every loc the kernels see is
a translated virtual id, so the whole read/write path differs from the static
pool: `translate_kv_loc_dense` for kv_indices and the cuda-graph write loc,
`HybridLinearKVPool._full_translate` for the model-level MLA entry points, and
page-envelope relocation on allocator compaction.

None of that is covered by the CPU/GPU unit tests, which pin the pool in
isolation. This is the end-to-end guard: accuracy must match the static-pool
baseline, and the prefix-cache branching case must still hit, since a radix hit
replays virtual locs whose physical pages may have moved under compaction.

Reference numbers on 2x H200 TP2, GSM8K 400 examples (2026-07-30):
static pools 0.915, `--enable-unified-memory` 0.900 (1 sigma ~= 0.015) -- both with
the attention backend pinned to triton, as this test runs it. For reference the
paged MLA kernels land in the same band on a single B300 TP1, GSM8K 200, unified
(2026-07-31): 0.915 with flashinfer prefill+decode, 0.900 with trtllm_mla. Those
are not exercised here (see the comment on `other_args`).
Nightly-only: it needs a second full 48B server launch, which is too much to add
to per-PR CI on top of the existing Kimi-Linear e2e coverage.

    python -m pytest test/registered/models_e2e/test_kimi_linear_unified_memory.py -v
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=600, suite="nightly-4-gpu", nightly=True)

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinearUnifiedMemory(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    # Same bar as the static-pool Kimi-Linear e2e test: unified memory must not
    # cost accuracy (measured 0.900 vs 0.915 static, see the module docstring).
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--enable-unified-memory",
        # Pinned because the resolved default is not portable: on pre-Blackwell
        # (this suite's runner is H100) an unspecified backend resolves to `fa3`,
        # which cannot read the dense views at all, so the un-pinned form fails at
        # startup with the page-major allowlist assertion. Unified memory on such a
        # host currently REQUIRES an explicit compatible --attention-backend; that
        # is a real usability gap, tracked separately, not something this test can
        # paper over.
        #
        # Consequence to keep in mind: pinning triton means this test does NOT
        # cover the paged MLA backends (trtllm_mla / flashinfer / cutedsl_mla /
        # tokenspeed_mla), which is where dense-id translation bugs live -- a
        # captured flashinfer decode reading untranslated virtual ids scored GSM8K
        # 0.000 on a healthy server. Those paths are covered by the unit tests plus
        # manual B300 runs; an sm100-gated case here would close the gap.
        "--attention-backend",
        "triton",
    ]


if __name__ == "__main__":
    unittest.main()
