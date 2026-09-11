"""DeepSeek-V4-Flash 8-GPU PDMux + HiCache (TP8).

Reuses the checks in test_pdmux_hicache.py against the model PDMux is actually
tuned for. Mirrors test/manual/dsv4/test_dsv4_flash_pdmux_sanity_tp8.py's launch
recipe and adds the hierarchical cache.

    SGLANG_TEST_DSV4_FLASH_MODEL_PATH=/model/DeepSeek-V4-Flash \
        python -m pytest test/manual/pdmux/test_dsv4_pdmux_hicache_tp8.py -v

Routing worth knowing before reading a failure: DSV4 is in the hybrid-SWA
architecture allowlist, so `--enable-hierarchical-cache` builds a
UnifiedRadixCache with the DeepSeek-V4 host stack (SWA + C4 + C4-indexer +
C128 sidecar pools), not the plain HiRadixCache. The indexer KV is anchored to
the KV pool with an all-pages hit policy, so a load-back restores index keys
along with the main KV -- which is what makes the greedy-output check below
meaningful rather than vacuous.

Not covered here: `--enable-hisparse`, DSV4's *sparse-attention* host tier. It
requires --disable-radix-cache, which --enable-hierarchical-cache forbids, so
the two can never run together and this file says nothing about that path.
"""

from __future__ import annotations

import os
import unittest

from sglang.test.test_utils import CustomTestCase

from test.manual.pdmux.test_pdmux_hicache import (
    SPLIT_FORWARD_TOKEN_BUDGET,
    PDMuxHiCacheMixin,
    sm_multiple,
)

DSV4_FLASH_MODEL_PATH = os.environ.get(
    "SGLANG_TEST_DSV4_FLASH_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_MHC_PREWARM": "0",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    # Pure-TP (no deepep A2A) resolves the auto MoE runner to Triton, which
    # cannot execute mxfp4-packed experts; dequant them to FP8 at load.
    "SGLANG_DSV4_FP4_DEQUANT": "1",
}
if not os.path.isdir(DSV4_FLASH_MODEL_PATH):
    DSV4_FLASH_ENV["SGLANG_DSV4_FP4_EXPERTS"] = "0"

# DSV4 forces page_size 256, so the prompt must be a whole number of pages for
# the prefix to be cacheable end to end.
DSV4_PROMPT_LEN = 4096
DSV4_MAX_TOTAL_TOKENS = 32768


class DSV4PDMuxHiCacheMixin(PDMuxHiCacheMixin):
    model_path = DSV4_FLASH_MODEL_PATH
    prompt_len = DSV4_PROMPT_LEN
    max_total_tokens = DSV4_MAX_TOTAL_TOKENS
    server_env = DSV4_FLASH_ENV
    extra_server_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        # PDMux splits by layer rather than by chunk here, matching the
        # existing DSV4 PDMux sanity run.
        "--chunked-prefill-size",
        "-1",
    ]


class TestDSV4PDMuxExclusivePartitionsHiCache(DSV4PDMuxHiCacheMixin, CustomTestCase):
    """Default layout: prefill and decode own disjoint green-context SM sets."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
        )


class TestDSV4PDMuxOverlappedMasksHiCache(DSV4PDMuxHiCacheMixin, CustomTestCase):
    """Overlapped layout: prefill stays capped, decode reaches every SM."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        import torch

        total_sm = torch.cuda.get_device_properties(0).multi_processor_count
        divisions = [
            (sm_multiple(total_sm // 4), 1),
            (sm_multiple(total_sm // 2), 8),
        ]
        entries = "".join(
            f"  - [{prefill_sm}, 0, {threshold}]\n"
            for prefill_sm, threshold in divisions
        )
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
            "overlap_decode_full_sm: true\n"
            f"manual_divisions:\n{entries}"
        )


class TestDSV4PDMuxExclusivePartitionsHiCacheStandardPrefill(
    TestDSV4PDMuxExclusivePartitionsHiCache
):
    """Exclusive layout, prefill submitted as one standard EXTEND."""

    pdmux_prefill_mode = "standard"


class TestDSV4PDMuxOverlappedMasksHiCacheStandardPrefill(
    TestDSV4PDMuxOverlappedMasksHiCache
):
    """Overlapped layout, prefill submitted as one standard EXTEND."""

    pdmux_prefill_mode = "standard"


if __name__ == "__main__":
    unittest.main()
