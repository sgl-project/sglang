"""SWA coverage for decode-side radix cache on gpt-oss-20b.

The decode worker reuses full-attention prefix KV while transferring the SWA
window fresh per request. This path requires the unified radix tree and validates
both multi-turn cache hits and two-pass GSM8K accuracy.
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

register_cuda_ci(est_time=365, stage="extra-b", runner_config="8-gpu-h200")

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
    gsm8k_min_score = 0.45
    # GPT-OSS needs chat formatting to avoid repetitive raw completions.
    gsm8k_eval_args = {
        "api": "chat",
        "reasoning_effort": "low",
        "max_tokens": 2048,
    }
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
