"""SWA-path coverage for the decode-side radix cache (unified radix tree).

gpt-oss-20b is a hybrid SWA model: full-attention layers reuse the decode radix
prefix while the sliding-window state is transferred fresh per turn. SWA +
decode-side radix cache is supported only via the unified radix tree, so the
servers launch with SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 (see extra_*_env below);
without it the decode server rejects the combination.

Reuses the registered decode-radix mixin (multi-turn cache-hit + 2-pass gsm8k).
gpt-oss-20b is cached on the 8-gpu-h20 CI runners (resolved via
try_cached_model in the mixin), so this runs in CI alongside the non-SWA
decode-radix coverage.
"""

import unittest

from test_disaggregation_decode_radix_cache import (
    DisaggregationDecodeRadixCacheTestMixin,
    _has_nixl,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE, is_in_ci

register_cuda_ci(est_time=600, stage="base-c", runner_config="8-gpu-h20")

SWA_SERVER_ARGS = ["--page-size", "64", "--attention-backend", "triton"]


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWANixl(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    # mxfp4 gpt-oss with the 512-token eval cap truncates reasoning, so absolute
    # gsm8k accuracy is low (~0.55), hence the lower min score. The second gsm8k
    # pass hits the decode radix cache (the reuse correctness check).
    gsm8k_min_score = 0.45
    # SWA + decode-side radix cache is gated to the unified radix tree.
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_prefill_args = SWA_SERVER_ARGS
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
