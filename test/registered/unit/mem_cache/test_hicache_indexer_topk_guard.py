"""Guard-compatibility tests for HiCache + capture flags (#26975).

INDEXER_TOPK is now allowed with --enable-hierarchical-cache (sidecar pool);
--enable-return-routed-experts stays guarded. Pure CPU arg-validation against a
stub -- no model load, no GPU.
"""

import types
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_args(**overrides) -> types.SimpleNamespace:
    """A stub carrying exactly the attributes _handle_cache_compatibility reads."""
    defaults = dict(
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        enable_return_routed_experts=False,
        enable_return_indexer_topk=False,
        disaggregation_decode_enable_offload_kvcache=False,
        disaggregation_mode="null",
        hicache_storage_backend=None,
        swa_full_tokens_ratio=1.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestHiCacheCaptureGuards(unittest.TestCase):
    def _run(self, **overrides):
        ServerArgs._handle_cache_compatibility(_make_args(**overrides))

    def test_indexer_topk_allowed_with_hicache(self):
        # The fix: this combination must no longer raise.
        self._run(enable_hierarchical_cache=True, enable_return_indexer_topk=True)

    def test_routed_experts_still_guarded(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(enable_hierarchical_cache=True, enable_return_routed_experts=True)
        self.assertIn("return-routed-experts", str(ctx.exception))

    def test_routed_experts_alone_ok(self):
        # Without hierarchical cache, routed-experts is fine.
        self._run(enable_hierarchical_cache=False, enable_return_routed_experts=True)

    def test_disable_radix_still_guarded(self):
        with self.assertRaises(ValueError):
            self._run(enable_hierarchical_cache=True, disable_radix_cache=True)


if __name__ == "__main__":
    unittest.main()
