import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVSender,
    TransferInfo,
)


class TestMooncakeDecodeRadixTransfer(unittest.TestCase):
    def test_transfer_info_preserves_decode_prefix_len(self):
        msg = [
            b"123",
            b"127.0.0.1",
            b"4567",
            b"session",
            np.array([10, 11], dtype=np.int32).tobytes(),
            b"3",
            np.array([7], dtype=np.int32).tobytes(),
            b"1",
            b"64",
        ]

        info = TransferInfo.from_zmq(msg)

        self.assertFalse(info.is_dummy)
        self.assertEqual(info.decode_prefix_len, 64)
        np.testing.assert_array_equal(
            info.dst_kv_indices, np.array([10, 11], dtype=np.int32)
        )
        self.assertEqual(info.dst_state_indices, [7])

    def test_empty_kv_with_aux_is_not_dummy_full_hit(self):
        msg = [
            b"123",
            b"127.0.0.1",
            b"4567",
            b"session",
            b"",
            b"3",
            b"",
            b"1",
            b"128",
        ]

        info = TransferInfo.from_zmq(msg)

        self.assertFalse(info.is_dummy)
        self.assertEqual(info.decode_prefix_len, 128)
        self.assertEqual(info.dst_kv_indices.size, 0)

    def test_sender_schedules_zero_page_last_chunk(self):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.bootstrap_room = 5
        sender.curr_idx = 0
        sender.num_kv_indices = 0
        sender.aux_index = 9
        sender.kv_mgr = SimpleNamespace(
            enable_all_cp_ranks_for_transfer=False,
            is_dummy_cp_rank=False,
            add_transfer_request=MagicMock(),
            req_to_decode_prefix_len={5: 32},
        )

        self.assertEqual(sender.pop_decode_prefix_len(), 32)
        self.assertTrue(sender.should_send_kv_chunk(num_pages=0, last_chunk=True))

        state_indices = [np.array([4], dtype=np.int32)]
        sender.send(np.array([], dtype=np.int32), state_indices=state_indices)

        sender.kv_mgr.add_transfer_request.assert_called_once()
        args = sender.kv_mgr.add_transfer_request.call_args.args
        kwargs = sender.kv_mgr.add_transfer_request.call_args.kwargs
        self.assertEqual(args[0], 5)
        self.assertEqual(args[2], slice(0, 0))
        self.assertTrue(args[3])
        self.assertEqual(kwargs["aux_index"], 9)
        self.assertIs(kwargs["state_indices"], state_indices)

    def test_prebuilt_batch_uses_delta_cache_locations_after_prefix(self):
        class FakeBatch(ScheduleBatchDisaggregationDecodeMixin):
            pass

        batch = FakeBatch()
        batch.device = "cpu"
        batch.return_logprob = False
        batch.model_config = SimpleNamespace(vocab_size=128)
        batch.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 20, 21]], dtype=torch.int64)
        )
        batch.reqs = [
            SimpleNamespace(
                req_pool_idx=0,
                fill_ids=[1, 2, 3, 4, 5],
                prefix_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
                extend_input_len=2,
                origin_input_ids=[1, 2, 3, 4, 5],
                output_ids=[],
                retracted_stain=False,
                cached_tokens=0,
                already_computed=0,
                is_retracted=True,
                extend_logprob_start_len=0,
                multimodal_inputs=None,
            )
        ]

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=SimpleNamespace(),
        ):
            batch.prepare_for_prebuilt()

        self.assertEqual(batch.out_cache_loc.tolist(), [20, 21])
        self.assertEqual(batch.input_ids.tolist(), [4, 5])
        self.assertEqual(batch.prefix_lens, [3])


if __name__ == "__main__":
    unittest.main()
