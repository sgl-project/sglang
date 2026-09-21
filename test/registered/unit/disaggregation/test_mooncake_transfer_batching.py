import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestMooncakeTransferBatching(unittest.TestCase):
    @staticmethod
    def _make_manager(
        side_effect=None, enable_custom_mem_pool=False, max_batch_indices=0
    ):
        engine = MagicMock()
        if side_effect is None:
            engine.batch_transfer_sync.return_value = 0
        else:
            engine.batch_transfer_sync.side_effect = side_effect
        manager = SimpleNamespace(
            engine=engine,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=1,
            enable_custom_mem_pool=enable_custom_mem_pool,
            custom_mem_pool_type="NVLINK" if enable_custom_mem_pool else None,
            enable_deferred_decode_kv_release=False,
            max_transfer_batch_indices=max_batch_indices,
            get_mla_kv_ptrs_with_pp=MagicMock(
                return_value=([1000, 2000], [5000, 6000], 2)
            ),
        )
        manager._transfer_data = lambda session, blocks: (
            MooncakeKVManager._transfer_data(manager, session, blocks)
        )
        manager._await_transfer_futures = lambda futures: (
            MooncakeKVManager._await_transfer_futures(manager, futures)
        )
        return manager

    @staticmethod
    def _send(
        manager,
        dst_device_data_indices=None,
        dst_device_data_ptrs=None,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000],
                dst_data_ptrs=[5000, 6000],
                item_lens=[10, 20],
                prefill_data_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                dst_data_indices=np.array([10, 11, 12, 13, 14], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=dst_device_data_indices,
                dst_device_data_ptrs=dst_device_data_ptrs,
            )

    def test_slices_index_arrays_before_forming_transfer_ranges(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6200], [20, 40]),
                call("session", [1020, 2040], [5120, 6240], [20, 40]),
                call("session", [1040, 2080], [5140, 6280], [10, 20]),
            ],
        )

    def test_preserves_legacy_single_batch_path_for_short_transfers(self):
        for max_batch_indices in (0, 5, 6):
            with self.subTest(max_batch_indices=max_batch_indices):
                manager = self._make_manager(max_batch_indices=max_batch_indices)
                ret = self._send(manager)

                self.assertEqual(ret, 0)
                manager.engine.batch_transfer_sync.assert_called_once_with(
                    "session",
                    [1000, 2000],
                    [5100, 6200],
                    [50, 100],
                )

    def test_stops_after_first_failed_index_batch(self):
        manager = self._make_manager(side_effect=[0, -1], max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, -1)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)

    def test_uses_device_page_indices_in_batched_path(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(
            manager,
            dst_device_data_indices=np.array([20, 21, 22, 23, 24], dtype=np.int32),
            dst_device_data_ptrs={6000},
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6400], [20, 40]),
                call("session", [1020, 2040], [5120, 6440], [20, 40]),
                call("session", [1040, 2080], [5140, 6480], [10, 20]),
            ],
        )

    def test_preserves_one_transfer_per_layer_for_custom_mem_pool(self):
        manager = self._make_manager(enable_custom_mem_pool=True, max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)
        manager.engine.batch_transfer_sync.assert_has_calls(
            [
                call("session", [1000], [5100], [50]),
                call("session", [2000], [6200], [100]),
            ],
            any_order=True,
        )


class TestDcpDraftHeadTransfer(unittest.TestCase):
    def test_transfers_draft_heads_to_logical_destination_rows(self):
        for src_tp, dst_tp in ((4, 8), (8, 4), (8, 8), (4, 32), (32, 4)):
            for custom_pool in (False, True):
                for batch_size in (0, 37):
                    with self.subTest(
                        src_tp=src_tp,
                        dst_tp=dst_tp,
                        custom_pool=custom_pool,
                        batch_size=batch_size,
                    ):
                        self._check_transfer(src_tp, dst_tp, custom_pool, batch_size)

    def test_rejects_pure_mla_with_unequal_draft_head_widths(self):
        for src_tp, dst_tp in ((4, 8), (8, 4)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                with self.assertRaisesRegex(ValueError, "dummy prefill senders"):
                    self._check_transfer(src_tp, dst_tp, False, 37, pure_mla=True)

    def test_sliced_draft_stops_after_failed_batch(self):
        self._check_transfer(4, 8, False, 37, fail_draft=True)

    def _check_transfer(
        self, src_tp, dst_tp, custom_pool, batch_size, fail_draft=False, pure_mla=False
    ):
        page_size, tokens, heads, head_bytes = 64, 249, 16, 4
        src_width, dst_width = (
            max(1, heads // src_tp) * head_bytes,
            max(1, heads // dst_tp) * head_bytes,
        )
        src_pages = np.array([1, 3, 4, 7], dtype=np.int32)
        logical = np.arange(tokens)
        src_rows = src_pages[logical // page_size] * page_size + logical % page_size
        expected = (
            np.arange(tokens * heads * head_bytes, dtype=np.int64)
            .reshape(tokens, heads, head_bytes)
            .astype(np.uint8)
        )
        for dst_rank in range(dst_tp):
            dst_buffers = {
                base: np.zeros(16384 * max(8, dst_width), dtype=np.uint8)
                for base in (1000000, 2000000, 3000000, 4000000)
            }
            source_ranks = (
                range(dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp)
                if src_tp >= dst_tp
                else [dst_rank * src_tp // dst_tp]
            )
            for src_rank in source_ranks:
                src_head_start = (src_rank // max(1, src_tp // heads)) * max(
                    1, heads // src_tp
                )
                source = np.zeros(1024 * src_width, dtype=np.uint8)
                source.reshape(-1, src_width)[src_rows] = expected[
                    :, src_head_start : src_head_start + max(1, heads // src_tp)
                ].reshape(tokens, src_width)
                target = np.zeros(1024 * 8, dtype=np.uint8)
                target.reshape(-1, 8)[src_rows] = (
                    np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)
                )
                src_buffers = {10000: target, 100000: source, 200000: source}

                failed_batches = []

                def transfer(
                    session, blocks, src_buffers=src_buffers, dst_buffers=dst_buffers
                ):
                    draft_blocks = [block for block in blocks if block[1] >= 3000000]
                    if fail_draft and draft_blocks:
                        failed_batches.append(draft_blocks)
                        return 17
                    if batch_size and src_width != dst_width:
                        self.assertLessEqual(
                            len(draft_blocks), batch_size * (1 if custom_pool else 2)
                        )
                    for src, dst, size in blocks:
                        src_base = max(base for base in src_buffers if base <= src)
                        dst_base = max(base for base in dst_buffers if base <= dst)
                        dst_buffers[dst_base][
                            dst - dst_base : dst - dst_base + size
                        ] = src_buffers[src_base][
                            src - src_base : src - src_base + size
                        ]
                    return 0

                manager = SimpleNamespace(
                    is_mla_backend=pure_mla,
                    kv_args=SimpleNamespace(
                        page_size=page_size,
                        kv_layer_ids=[47, 93, 93],
                        kv_data_ptrs=[10000, 100000, 200000],
                        num_draft_entries=2,
                        engine_rank=src_rank + 2 * src_tp,
                    ),
                    attn_tp_size=src_tp,
                    max_transfer_batch_indices=batch_size,
                    enable_custom_mem_pool=custom_pool,
                    _transfer_data=transfer,
                    _await_transfer_futures=lambda futures: max(
                        f.result() for f in futures
                    ),
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = MooncakeKVManager.send_kvcache_dcp(
                        manager,
                        "session",
                        src_pages,
                        [1000000, 2000000, 3000000, 4000000],
                        np.array([2], dtype=np.int32),
                        dcp_token_item_lens=[8, src_width, src_width],
                        dst_dcp_size=dst_tp,
                        dst_dcp_rank=dst_rank,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=tokens,
                        executor=executor,
                        dst_layer_ids=[3, 47, 93, 93],
                        dst_kv_item_lens=[
                            page_size * 8,
                            page_size * 8,
                            page_size * dst_tp * dst_width,
                            page_size * dst_tp * dst_width,
                        ],
                        dst_tp_rank=dst_rank,
                        dst_attn_tp_size=dst_tp,
                    )
                if fail_draft:
                    self.assertEqual(result, 17)
                    self.assertEqual(len(failed_batches), 1)
                    return
                self.assertEqual(result, 0)
            dst_head_start = (dst_rank // max(1, dst_tp // heads)) * max(
                1, heads // dst_tp
            )
            for base in (3000000, 4000000):
                actual = dst_buffers[base].reshape(-1, dst_width)[
                    2 * page_size * dst_tp + logical
                ]
                np.testing.assert_array_equal(
                    actual,
                    expected[
                        :,
                        dst_head_start : dst_head_start + max(1, heads // dst_tp),
                    ].reshape(tokens, dst_width),
                )
            owned = np.arange(dst_rank, tokens, dst_tp)
            actual_target = dst_buffers[2000000].reshape(-1, 8)[
                2 * page_size + owned // dst_tp
            ]
            np.testing.assert_array_equal(
                actual_target,
                np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)[owned],
            )
            self.assertFalse(dst_buffers[1000000].any())


if __name__ == "__main__":
    unittest.main()
