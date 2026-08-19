"""Kimi-Linear (MLA full attention + KDA linear attention) served from the
unified memory pool.

Under `--enable-unified-memory` the MLA full side is exposed as DENSE per-layer
views and every loc the kernels see is a translated virtual id, so the whole
read/write path differs from the static pool. The unit tests pin that pool in
isolation; this is the end-to-end guard. `test_prefix_cache_branching` carries
most of the weight: a radix hit replays virtual locs whose physical pages may
have moved under compaction.

No `--attention-backend` is pinned on purpose -- the test runs whatever the host
resolves to (`fa3` on this suite's H100 runner, also the H200 default). Both
defects found in review on #32972 were reachable only under a resolved default,
which a pinned test hides by construction.

Reference GSM8K, all with `--enable-unified-memory`:
  - 2x H200 TP2, resolved default (fa3): 0.917 @400, vs 0.915 static (1 sigma
    ~= 0.015). This file as written scores 0.920 @200.
  - 2x H200 TP2, `--attention-backend triton`: 0.900 @400.
  - 1x B300 TP1: 0.915 flashinfer, 0.900 trtllm_mla, @200.

Nightly-only: a second full 48B server launch is too much for per-PR CI on top
of the existing Kimi-Linear e2e coverage.

    python -m pytest test/registered/models_e2e/test_kimi_linear_unified_memory.py -v
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=570, stage="nightly", runner_config="4-gpu-h100")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinearUnifiedMemory(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    # Same bar as the static-pool Kimi-Linear e2e test: unified memory must not
    # cost accuracy (measured 0.917 vs 0.915 static, see the module docstring).
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--enable-unified-memory",
    ]


if __name__ == "__main__":
    unittest.main()
