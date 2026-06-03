"""SWA-path coverage for the decode-side radix cache.

gpt-oss-20b is a hybrid SWA model: full-attention layers reuse the decode
radix prefix while the sliding-window state is transferred fresh per turn.
This file is registered `disabled` so it does not run in scheduled CI;
trigger it manually on a PR with:

    /rerun-test test/registered/disaggregation/test_disaggregation_decode_radix_cache_swa.py
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
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    is_in_ci,
)

register_cuda_ci(
    est_time=600,
    stage="base-c",
    runner_config="8-gpu-h20",
    disabled="manual-only SWA-path coverage; run via /rerun-test",
)

# SWA tail prealloc requires page_size > 1 plus an allocator exposing
# alloc_extend_swa_tail; triton is the validated attention backend for the
# gpt-oss hybrid SWA config.
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
    # mxfp4 gpt-oss with the 512-token eval cap truncates reasoning, so
    # absolute gsm8k accuracy is low (~0.55). Single pass to keep the manual
    # run short; multi-turn cache reuse is covered by the cache-hit test.
    gsm8k_min_score = 0.45
    gsm8k_num_passes = 1
    extra_prefill_args = SWA_SERVER_ARGS
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
