import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    release_kv_cache,
    retraction_backup,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.kv_cache_builder import _supports_host_pool_retraction
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4RetractionBackupCapability(unittest.TestCase):
    @staticmethod
    def _make_dsv4_pool(
        *,
        with_separate_swa_pool: bool,
        hisparse: bool = False,
    ):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = object() if with_separate_swa_pool else None
        pool.c4_kv_pool = object.__new__(HiSparseC4DevicePool) if hisparse else object()
        return pool

    def test_dsv4_separate_swa_pool_selects_host_pool(self):
        pool = self._make_dsv4_pool(with_separate_swa_pool=True)
        self.assertIsInstance(pool, BaseSWAKVPool)
        self.assertTrue(_supports_host_pool_retraction(pool, 32))
        self.assertFalse(_supports_host_pool_retraction(pool, 0))

    def test_mha_pool_selection_is_unchanged(self):
        pool = object.__new__(MHATokenToKVPool)
        self.assertTrue(_supports_host_pool_retraction(pool, None))

    def test_standard_swa_pool_selection_is_unchanged(self):
        pool = object.__new__(SWAKVPool)
        pool.swa_kv_pool = object()
        self.assertTrue(_supports_host_pool_retraction(pool, 32))

    def test_dsv4_unified_pool_does_not_select_unsupported_host_path(self):
        pool = self._make_dsv4_pool(with_separate_swa_pool=False)
        self.assertFalse(_supports_host_pool_retraction(pool, 32))

    def test_dsv4_hisparse_does_not_select_incomplete_host_path(self):
        pool = self._make_dsv4_pool(with_separate_swa_pool=True, hisparse=True)
        self.assertFalse(_supports_host_pool_retraction(pool, 32))

    def test_dsv4_speculative_state_fails_closed(self):
        pool = self._make_dsv4_pool(with_separate_swa_pool=True)
        self.assertFalse(_supports_host_pool_retraction(pool, 32, True))

    @patch("sglang.srt.mem_cache.unified_radix_cache.get_spec")
    def test_unified_cache_accepts_dsv4_host_pool_group(self, mock_get_spec):
        mock_get_spec.return_value = SimpleNamespace(speculative_algorithm=None)
        pool = self._make_dsv4_pool(with_separate_swa_pool=True)
        cache = self._make_cache(pool)
        self.assertTrue(cache.supports_retraction_backup())

    @patch("sglang.srt.mem_cache.unified_radix_cache.get_spec")
    def test_unified_cache_rejects_incomplete_hisparse_host_group(self, mock_get_spec):
        mock_get_spec.return_value = SimpleNamespace(speculative_algorithm=None)
        pool = self._make_dsv4_pool(with_separate_swa_pool=True, hisparse=True)
        cache = self._make_cache(pool)
        self.assertFalse(cache.supports_retraction_backup())

    @staticmethod
    def _make_cache(pool):
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = object()
        cache.host_pool_group = SimpleNamespace(
            entry_map={PoolName.KV: object(), PoolName.SWA: object()}
        )
        cache.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: pool)
        cache.supports_mamba = lambda: False
        cache.supports_swa = lambda: True
        return cache

    def test_unsupported_cpu_snapshot_aborts_request_without_escaping(self):
        class UnsupportedReq:
            rid = "unsupported"

            @staticmethod
            def offload_kv_cache(*_args):
                raise NotImplementedError

        req = UnsupportedReq()
        self.assertFalse(
            retraction_backup(
                req,
                tree_cache=SimpleNamespace(),
                req_to_token_pool=SimpleNamespace(),
                token_to_kv_pool_allocator=SimpleNamespace(),
                backend="cpu_tensor",
            )
        )
        self.assertIsNone(req.retraction_backup)

    @patch("sglang.srt.mem_cache.common.get_serving")
    @patch("sglang.srt.mem_cache.common.get_spec")
    def test_restore_failure_releases_kv_without_a_radix_lock(
        self, mock_get_spec, mock_get_serving
    ):
        mock_get_spec.return_value = SimpleNamespace(speculative_algorithm=None)
        mock_get_serving.return_value = SimpleNamespace(strip_thinking_cache=False)

        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=4),
            kv_committed_len=4,
            cache_protected_len=0,
            last_node=None,
            swa_uuid_for_lock=None,
            swa_prefix_lock_released=False,
            skip_lock_node_ids={},
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[],
            effective_kv_committed_len=lambda: 4,
        )
        allocator = SimpleNamespace(page_size=1, free_segment=MagicMock())

        def free_req_slot(finished_req):
            finished_req.req_pool_idx = None

        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[11, 12, 13, 14]]),
            free=MagicMock(side_effect=free_req_slot),
        )
        tree_core = SimpleNamespace(dec_lock_ref=MagicMock(side_effect=KeyError(None)))
        cache = object.__new__(UnifiedRadixCache)
        cache.session = SimpleNamespace(
            try_cache_finished_req=lambda *_args, **_kwargs: False,
            try_dec_lock_ref=lambda *_args, **_kwargs: None,
        )
        cache.disable = False
        cache.tree_core = tree_core
        cache.req_to_token_pool = req_to_token_pool
        cache.token_to_kv_pool_allocator = allocator
        cache._components_tuple = ()
        cache.enable_session_radix_cache = False
        cache.supports_mamba = lambda: False

        release_kv_cache(req, cache, is_insert=False)

        allocator.free_segment.assert_called_once()
        tree_core.dec_lock_ref.assert_not_called()
        req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)

    def test_retraction_uses_live_swa_frontier_for_unaligned_sequence(self):
        page_size = 256
        full_page_start = 1024
        swa_page_start = 2048
        num_tokens = 300
        full_indices = torch.arange(
            full_page_start, full_page_start + num_tokens, dtype=torch.int64
        )

        def translate_live_swa(indices):
            logical = indices - full_page_start
            translated = torch.zeros_like(logical)
            live = logical >= page_size
            translated[live] = swa_page_start + logical[live] - page_size
            return translated

        cache = self._make_retraction_cache(
            full_indices=full_indices,
            translate_live_swa=translate_live_swa,
            page_size=page_size,
            sliding_window_size=128,
        )
        req = SimpleNamespace(
            rid="unaligned",
            req_pool_idx=0,
            seqlen=num_tokens + 1,
            kv=SimpleNamespace(swa_evicted_seqlen=page_size),
        )

        full_padded, transfers = cache._retraction_device_transfers(req)
        self.assertEqual(len(full_padded), 2 * page_size)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.SWA)
        self.assertEqual(len(transfers[0].device_indices), page_size)
        self.assertEqual(int(transfers[0].device_indices[0]), swa_page_start)
        self.assertEqual(
            int(transfers[0].device_indices[-1]), swa_page_start + page_size - 1
        )
        with self.assertRaisesRegex(RuntimeError, "SWA window changed"):
            cache._retraction_device_transfers(req, expected_swa_window_start=0)

    def test_unmapped_live_swa_aborts_backup_without_asserting(self):
        page_size = 4
        full_indices = torch.arange(8, 14, dtype=torch.int64)
        cache = self._make_retraction_cache(
            full_indices=full_indices,
            translate_live_swa=lambda indices: torch.zeros_like(indices),
            page_size=page_size,
            sliding_window_size=4,
        )
        req = SimpleNamespace(
            rid="unmapped",
            req_pool_idx=0,
            seqlen=len(full_indices) + 1,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
        )

        cache.host_pool_group = SimpleNamespace()
        self.assertIsNone(cache.retraction_backup(req))

    def test_padding_never_crosses_into_the_next_device_page(self):
        self.assertTrue(
            torch.equal(
                UnifiedRadixCache._pad_retraction_indices(torch.tensor([512, 513]), 4),
                torch.tensor([512, 513, 514, 515]),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "device-page boundary"):
            UnifiedRadixCache._pad_retraction_indices(torch.tensor([513, 514]), 4)

    def test_restore_rejects_extra_pool_length_mismatch(self):
        cache = object.__new__(UnifiedRadixCache)
        cache._retraction_device_transfers = lambda _req, **_kwargs: (
            torch.arange(4),
            [PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(4))],
        )
        cache.cache_controller = SimpleNamespace(
            _resolve_pool_transfers_allocation=lambda transfers, **_kwargs: transfers
        )
        cache.retraction_discard = MagicMock()
        backup = RetractionBackup(
            host_indices=torch.arange(4),
            pool_transfers=[
                PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(8))
            ],
            swa_window_start=0,
        )

        self.assertFalse(
            cache.retraction_restore(SimpleNamespace(rid="mismatch"), backup)
        )
        cache.retraction_discard.assert_called_once_with(backup)

    @staticmethod
    def _make_retraction_cache(
        *, full_indices, translate_live_swa, page_size, sliding_window_size
    ):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=page_size)
        cache.page_size = page_size
        cache._sliding_window_size = sliding_window_size
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=full_indices.unsqueeze(0)
        )
        kv_cache = SimpleNamespace(
            swa_kv_pool=SimpleNamespace(page_size=page_size),
            translate_loc_from_full_to_swa=translate_live_swa,
        )
        cache.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: kv_cache)
        cache.supports_swa = lambda: True
        cache.sidecar_pool_specs = []
        return cache


if __name__ == "__main__":
    unittest.main()
