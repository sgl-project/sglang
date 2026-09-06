import threading
import time
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ImmediateExecutor:
    def __init__(self):
        self.submit_count = 0

    def submit(self, fn, *args):
        self.submit_count += 1
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as e:
            future.set_exception(e)
        return future


class TestMooncakeCustomMemPoolBatch(CustomTestCase):
    def test_wait_for_transfer_rooms_wakes_on_terminal_status(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: KVPoll.Transferring, 8: KVPoll.Transferring}
        manager._request_status_lock = threading.RLock()
        manager._transfer_completion_condition = threading.Condition()

        def complete_rooms():
            time.sleep(0.01)
            manager.update_status(7, KVPoll.Success)
            manager.update_status(8, KVPoll.Success)

        thread = threading.Thread(target=complete_rooms)
        thread.start()
        try:
            self.assertTrue(manager.wait_for_transfer_rooms({7, 8}, 0.2))
        finally:
            thread.join()

    def test_wait_for_transfer_rooms_is_bounded(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: KVPoll.Transferring}
        manager._request_status_lock = threading.RLock()
        manager._transfer_completion_condition = threading.Condition()

        start_time = time.perf_counter()
        self.assertFalse(manager.wait_for_transfer_rooms({7}, 0.01))
        self.assertLess(time.perf_counter() - start_time, 0.1)

    def test_wait_for_transfer_rooms_treats_cleared_room_as_terminal(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {}
        manager._transfer_completion_condition = threading.Condition()

        self.assertTrue(manager.wait_for_transfer_rooms({7}, 0.01))

    def test_generic_intra_node_nvlink_combines_all_layers(self):
        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "INTRA_NODE_NVLINK"
        manager.pp_size = 1
        manager._transfer_data = MagicMock(return_value=0)
        executor = MagicMock()

        layer_count = 43
        src_ptrs = [100_000 + i * 10_000 for i in range(layer_count)]
        dst_ptrs = [200_000 + i * 10_000 for i in range(layer_count)]
        item_lens = [100] * layer_count

        ret = manager._send_kvcache_generic(
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([1, 2, 5], dtype=np.int32),
            dst_data_indices=np.array([11, 12, 15], dtype=np.int32),
            executor=executor,
        )

        self.assertEqual(ret, 0)
        executor.submit.assert_not_called()
        manager._transfer_data.assert_called_once()
        session_id, blocks = manager._transfer_data.call_args.args
        self.assertEqual(session_id, "session")
        self.assertEqual(len(blocks), layer_count * 2)
        self.assertEqual(
            blocks[:2],
            [
                (src_ptrs[0] + 100, dst_ptrs[0] + 1_100, 200),
                (src_ptrs[0] + 500, dst_ptrs[0] + 1_500, 100),
            ],
        )
        self.assertEqual(
            blocks[-2:],
            [
                (src_ptrs[-1] + 100, dst_ptrs[-1] + 1_100, 200),
                (src_ptrs[-1] + 500, dst_ptrs[-1] + 1_500, 100),
            ],
        )

    def test_dcp_intra_node_nvlink_combines_all_layers(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "INTRA_NODE_NVLINK"
        manager.kv_args = MagicMock(
            page_size=256,
            kv_data_ptrs=[100_000 + i * 10_000 for i in range(43)],
            kv_layer_ids=list(range(43)),
        )
        manager._transfer_data = MagicMock(return_value=0)
        executor = MagicMock()
        dst_ptrs = [200_000 + i * 10_000 for i in range(43)]

        ret = manager.send_kvcache_dcp(
            mooncake_session_id="session",
            prefill_kv_indices=np.array([2], dtype=np.int32),
            dst_kv_ptrs=dst_ptrs,
            dst_kv_indices=np.array([5], dtype=np.int32),
            dcp_token_item_lens=[16] * 43,
            dst_dcp_size=1,
            dst_dcp_rank=0,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=256,
            executor=executor,
            dst_layer_ids=list(range(43)),
        )

        self.assertEqual(ret, 0)
        executor.submit.assert_not_called()
        manager._transfer_data.assert_called_once()
        session_id, blocks = manager._transfer_data.call_args.args
        self.assertEqual(session_id, "session")
        self.assertEqual(len(blocks), 43)
        self.assertEqual(
            blocks[0],
            (100_000 + 2 * 256 * 16, 200_000 + 5 * 256 * 16, 256 * 16),
        )
        self.assertEqual(
            blocks[-1],
            (
                100_000 + 42 * 10_000 + 2 * 256 * 16,
                200_000 + 42 * 10_000 + 5 * 256 * 16,
                256 * 16,
            ),
        )

    def test_generic_unverified_custom_pool_keeps_per_layer_submission(self):
        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "NVLINK"
        manager.pp_size = 1
        manager.enable_deferred_decode_kv_release = False
        manager._transfer_data = MagicMock(return_value=0)
        executor = ImmediateExecutor()

        ret = manager._send_kvcache_generic(
            mooncake_session_id="session",
            src_data_ptrs=[100_000, 110_000, 120_000],
            dst_data_ptrs=[200_000, 210_000, 220_000],
            item_lens=[100, 100, 100],
            prefill_data_indices=np.array([1, 2], dtype=np.int32),
            dst_data_indices=np.array([11, 12], dtype=np.int32),
            executor=executor,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(executor.submit_count, 3)
        self.assertEqual(manager._transfer_data.call_count, 3)

    def test_dcp_unverified_custom_pool_keeps_per_layer_submission(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "BAREX"
        manager.enable_deferred_decode_kv_release = False
        manager.kv_args = MagicMock(
            page_size=256,
            kv_data_ptrs=[100_000, 110_000, 120_000],
            kv_layer_ids=[0, 1, 2],
        )
        manager._transfer_data = MagicMock(return_value=0)
        executor = ImmediateExecutor()

        ret = manager.send_kvcache_dcp(
            mooncake_session_id="session",
            prefill_kv_indices=np.array([2], dtype=np.int32),
            dst_kv_ptrs=[200_000, 210_000, 220_000],
            dst_kv_indices=np.array([5], dtype=np.int32),
            dcp_token_item_lens=[16, 16, 16],
            dst_dcp_size=1,
            dst_dcp_rank=0,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=256,
            executor=executor,
            dst_layer_ids=[0, 1, 2],
        )

        self.assertEqual(ret, 0)
        self.assertEqual(executor.submit_count, 3)
        self.assertEqual(manager._transfer_data.call_count, 3)


if __name__ == "__main__":
    unittest.main()
