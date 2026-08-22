"""DSV4 coverage for decode-side radix cache on DeepSeek-V4-Flash.

CUDA DSV4 rides the same [FULL, SWA] unified-tree path as gpt-oss: c4/c128/
indexer rows derive positionally from the virtual full-id space, so FULL
prefix reuse keeps every compressed pool coherent. Non-spec launch only --
decode radix cache rejects speculative decoding globally.
"""

import unittest

from test_disaggregation_decode_radix_cache import (
    DisaggregationDecodeRadixCacheTestMixin,
    _has_mooncake,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import is_in_ci

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}

DSV4_SERVER_ARGS = [
    "--page-size",
    "256",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.9",
    "--skip-server-warmup",
    "--watchdog-timeout",
    "900",
]


@unittest.skipUnless(
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for DSV4 decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWADSV4(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "mooncake"
    model_name = DSV4_FLASH_MODEL
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    # Window 128 < page 256: prompts below window + page (512) reuse nothing,
    # and every hit loses up to 512 tokens to the SWA prefix cap, so the
    # cache-hit rounds need headroom above the kit's expected-cache floor.
    cache_hit_request_length = 1024
    two_pass_num_examples = 200
    gsm8k_min_score = 0.90
    # SWA + decode-side radix cache is gated to the unified radix tree.
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1", **DSV4_FLASH_ENV}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1", **DSV4_FLASH_ENV}
    extra_prefill_args = DSV4_SERVER_ARGS
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *DSV4_SERVER_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
