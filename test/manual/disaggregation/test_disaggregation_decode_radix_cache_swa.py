"""Manual SWA-path coverage for the decode-side radix cache.

gpt-oss-20b is a hybrid SWA model: full-attention layers reuse the decode
radix prefix while the sliding-window state is transferred fresh per turn.

This test lives under test/manual/ on purpose: it needs gpt-oss-20b, which the
CI runner pools do not cache and cannot fetch, so it is NOT registered for CI
(no register_*_ci call) and is never auto-discovered by run_suite.py. Run it by
hand on a box that has gpt-oss-20b cached and >= 2 GPUs (prefill GPU0 /
decode GPU1), e.g. the 2x L40S dev box:

    cd test/ && python3 manual/disaggregation/test_disaggregation_decode_radix_cache_swa.py
"""

import os
import sys
import unittest

# Reuse the registered disaggregation mixin without registering this file for
# CI. The mixin lives in test/registered/disaggregation/; add that dir to the
# path so the sibling import resolves when this file is run standalone.
_REGISTERED_DISAGG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "registered", "disaggregation")
)
if _REGISTERED_DISAGG not in sys.path:
    sys.path.insert(0, _REGISTERED_DISAGG)

from test_disaggregation_decode_radix_cache import (  # noqa: E402
    DisaggregationDecodeRadixCacheTestMixin,
    _has_nixl,
)

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

# SWA tail prealloc requires page_size > 1 plus an allocator exposing
# alloc_extend_swa_tail; triton is the validated attention backend for the
# gpt-oss hybrid SWA config.
SWA_SERVER_ARGS = ["--page-size", "64", "--attention-backend", "triton"]


@unittest.skipUnless(
    _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWANixl(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    # mxfp4 gpt-oss with the 512-token eval cap truncates reasoning, so
    # absolute gsm8k accuracy is low (~0.55), hence the lower min score. The
    # second gsm8k pass hits the decode radix cache (the reuse correctness check).
    gsm8k_min_score = 0.45
    extra_prefill_args = SWA_SERVER_ARGS
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
