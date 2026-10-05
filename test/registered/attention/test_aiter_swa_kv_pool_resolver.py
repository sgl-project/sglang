"""PR3: aiter SWA KV-pool resolver + paged-decode sliding-window guard.

``_resolve_swa_kv_pool`` returns the active pool for every non-draft worker
(behaviour-preserving), skips the target SWA mapping for EAGLE draft workers
(which own a separate draft pool), and falls back to the allocator's KV cache
for FROZEN_KV MTP. ``_reject_paged_decode_sliding_window`` fails loudly when a
sliding-window layer would reach ``paged_attention_ragged`` (which has no window
argument) instead of silently returning full-context (wrong) results.

Both are pure branching logic, exercised here with mocks (no GPU kernel).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _model_runner(active_pool, *, is_draft=False, frozen_kv_mtp=False, alloc_pool=None):
    return SimpleNamespace(
        token_to_kv_pool=active_pool,
        is_draft_worker=is_draft,
        spec_algorithm=SimpleNamespace(is_frozen_kv_mtp=lambda: frozen_kv_mtp),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: alloc_pool),
    )


class TestResolveSwaKvPool(CustomTestCase):
    def test_active_pool_is_swa_returns_it(self):
        swa = MagicMock(spec=SWAKVPool)
        mr = _model_runner(swa)
        self.assertIs(AiterAttnBackend._resolve_swa_kv_pool(mr), swa)

    def test_non_draft_non_swa_active_falls_back_to_allocator(self):
        alloc_swa = MagicMock(spec=SWAKVPool)
        mr = _model_runner(object(), alloc_pool=alloc_swa)
        self.assertIs(AiterAttnBackend._resolve_swa_kv_pool(mr), alloc_swa)

    def test_non_draft_no_swa_anywhere_returns_none(self):
        mr = _model_runner(object(), alloc_pool=object())
        self.assertIsNone(AiterAttnBackend._resolve_swa_kv_pool(mr))

    def test_eagle_draft_worker_skips_target_mapping(self):
        # Draft worker that is NOT frozen-KV MTP owns its own draft pool -> None,
        # even if the allocator holds an SWA pool.
        mr = _model_runner(
            object(),
            is_draft=True,
            frozen_kv_mtp=False,
            alloc_pool=MagicMock(spec=SWAKVPool),
        )
        self.assertIsNone(AiterAttnBackend._resolve_swa_kv_pool(mr))

    def test_frozen_kv_mtp_draft_worker_uses_allocator_pool(self):
        alloc_swa = MagicMock(spec=SWAKVPool)
        mr = _model_runner(
            object(), is_draft=True, frozen_kv_mtp=True, alloc_pool=alloc_swa
        )
        self.assertIs(AiterAttnBackend._resolve_swa_kv_pool(mr), alloc_swa)


class TestPagedDecodeSlidingWindowGuard(CustomTestCase):
    def test_raises_for_sliding_window_layer(self):
        layer = SimpleNamespace(sliding_window_size=1024, layer_id=7)
        with self.assertRaises(ValueError) as ctx:
            AiterAttnBackend._reject_paged_decode_sliding_window(layer)
        msg = str(ctx.exception)
        self.assertIn("sliding-window", msg)
        self.assertIn("SGLANG_USE_AITER_UNIFIED_ATTN=1", msg)
        self.assertIn("layer 7", msg)

    def test_inert_when_window_unset(self):
        layer = SimpleNamespace(sliding_window_size=None, layer_id=0)
        AiterAttnBackend._reject_paged_decode_sliding_window(layer)

    def test_inert_when_window_is_negative_one(self):
        layer = SimpleNamespace(sliding_window_size=-1, layer_id=0)
        AiterAttnBackend._reject_paged_decode_sliding_window(layer)


if __name__ == "__main__":
    unittest.main()
