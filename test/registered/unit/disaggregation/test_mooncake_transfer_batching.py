import concurrent.futures
import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


class TestMiniMaxStateTransfer(CustomTestCase):
    def test_index_truncates_but_dense_rejects_mismatched_page_lists(self):
        """Legacy index transfers copy the common prefix; incomplete dense KV must fail."""

        def copy_bytes(session, sources, destinations, lengths):
            for src, dst, length in zip(sources, destinations, lengths, strict=True):
                ctypes.memmove(dst, src, length)
            return 0

        for state in (StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV):
            for src_pages, dst_pages in (([1], [0]), ([1, 2], [0]), ([1], [0, 2])):
                with self.subTest(state=state, src=src_pages, dst=dst_pages):
                    src = np.arange(3, dtype=np.int32)
                    dst = np.full(3, -1, dtype=np.int32)
                    manager = MooncakeKVManager.__new__(MooncakeKVManager)
                    manager.kv_args = SimpleNamespace(
                        state_types=[state],
                        state_data_ptrs=[[src.ctypes.data]],
                        state_item_lens=[[src.itemsize]],
                        state_dim_per_tensor=[[]],
                        state_layer_ids=[[]],
                    )
                    manager.engine = SimpleNamespace(batch_transfer_sync=copy_bytes)
                    manager.pp_size = manager.attn_tp_size = 1
                    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                    manager.enable_custom_mem_pool = False
                    manager.max_transfer_batch_indices = 0
                    peer = SimpleNamespace(
                        dst_state_data_ptrs=[[dst.ctypes.data]],
                        dst_state_item_lens=[[dst.itemsize]],
                        dst_state_dim_per_tensor=[[]],
                        dst_state_layer_ids=[[]],
                        dst_attn_tp_size=1,
                    )
                    kwargs = dict(
                        req=SimpleNamespace(
                            mooncake_session_id="cpu", dst_state_indices=[dst_pages]
                        ),
                        prefill_state_indices=[src_pages],
                        executor=None,
                        target_rank_registration_info=peer,
                    )
                    if state == StateType.MINIMAX_DENSE_KV and len(src_pages) != len(
                        dst_pages
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "state index length mismatch"
                        ):
                            manager.maybe_send_extra(**kwargs)
                        np.testing.assert_array_equal(dst, [-1, -1, -1])
                    else:
                        self.assertEqual(manager.maybe_send_extra(**kwargs), 0)
                        np.testing.assert_array_equal(dst, [1, -1, -1])


if __name__ == "__main__":
    unittest.main()
