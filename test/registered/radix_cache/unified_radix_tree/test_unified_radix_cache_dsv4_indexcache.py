"""DeepSeek V4 Flash HiCache + UnifiedRadixCache with IndexCache.

Subclasses TestUnifiedDeepSeekV4FlashHiCache (the production Hierarchical Cache
recipe: --enable-hierarchical-cache, --hicache-write-policy write_through,
--hicache-mem-layout, SGLANG_ENABLE_UNIFIED_RADIX_TREE=1, compressed attention,
page-size 256, chunked prefill) and adds the DeepSeek V4 IndexCache override
(index_topk_freq=4).

This is the HiCache combination the plan asks for (distinct from HiSparse,
which is decode-side C4 sparse-page offload): it inherits the mixin's cold/hit,
partial-prefix-hit, and cached-token assertions so IndexCache is validated
against real HiCache write-through + unified radix tree behavior.

The base test class is referenced via the module (not imported into this
module's namespace) so unittest does NOT re-collect and re-run the base test's
heavy methods here -- only the IndexCache subclass runs.

Registry: extra-b, 4x H100.
"""

import unittest

import test_unified_radix_cache_kl_dsv4 as _dsv4_hicache

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1000, stage="extra-b", runner_config="4-gpu-h100")


class TestUnifiedDeepSeekV4FlashHiCacheIndexCache(
    _dsv4_hicache.TestUnifiedDeepSeekV4FlashHiCache
):
    """HiCache + UnifiedRadixCache with IndexCache freq=4."""

    @classmethod
    def _server_args(cls):
        # Reuse the full HiCache + unified-radix recipe, add the IndexCache
        # override so the C4 producer/shared paths run under HiCache
        # write-through + cold/hit + prefix-hit.
        return super()._server_args() + [
            "--json-model-override-args",
            '{"index_topk_freq": 4}',
        ]


if __name__ == "__main__":
    unittest.main()
