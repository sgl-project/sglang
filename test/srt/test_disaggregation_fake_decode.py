import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.scheduler import Scheduler


class TestFakeDecodePrealloc(unittest.TestCase):
    def test_fake_transfer_accepts_missing_bootstrap_host(self):
        class Receiver:
            def __init__(self, mgr, bootstrap_addr, bootstrap_room=None):
                self.bootstrap_addr = bootstrap_addr
                self.bootstrap_room = bootstrap_room

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disaggregation_transfer_backend="fake")
        )
        queue.transfer_backend = object()
        queue.kv_manager = object()
        queue.queue = []

        req = types.SimpleNamespace(
            rid="req",
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=123,
        )

        with patch(
            "sglang.srt.disaggregation.decode.get_kv_class", return_value=Receiver
        ):
            decode_req = DecodePreallocQueue._create_receiver_and_enqueue(queue, req)

        self.assertIs(decode_req.req, req)
        self.assertEqual(decode_req.kv_receiver.bootstrap_addr, "2.2.2.2:0")
        self.assertEqual(decode_req.kv_receiver.bootstrap_room, 123)
        self.assertEqual(queue.queue, [decode_req])


class TestFakeDecodeMigrationBootstrap(unittest.TestCase):
    def test_fake_backend_does_not_start_bootstrap_server(self):
        tokenizer_manager = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="fake",
                host="127.0.0.1",
                disaggregation_bootstrap_port=39000,
            )
        )

        with patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_kv_class"
        ) as get_kv_class:
            TokenizerControlMixin._ensure_pd_flip_migration_bootstrap_server(
                tokenizer_manager
            )

        get_kv_class.assert_not_called()
        self.assertFalse(hasattr(tokenizer_manager, "pd_flip_migration_bootstrap_server"))


class TestFakePDFlipTargetMigration(unittest.TestCase):
    def test_fake_target_receiver_init_skips_parallel_info_lookup(self):
        class Receiver:
            def __init__(self):
                self.init_rank = None

            def init(self, prefill_dp_rank):
                self.init_rank = prefill_dp_rank

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.transfer_backend = TransferBackend.FAKE
        scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            kv_manager=object()
        )

        receiver = Receiver()
        decode_req = types.SimpleNamespace(
            req=types.SimpleNamespace(
                bootstrap_host="127.0.0.1",
                bootstrap_port=39000,
                bootstrap_room=123,
            ),
            kv_receiver=receiver,
        )

        self.assertTrue(Scheduler._pd_flip_target_init_receiver(scheduler, decode_req))
        self.assertEqual(receiver.init_rank, 0)

    def test_fake_target_metadata_send_marks_metadata_ready(self):
        class Allocator:
            def alloc(self):
                return 0

        class Receiver:
            def __init__(self):
                self.sent = False

            def send_metadata(
                self, page_indices, metadata_index, state_indices, decode_prefix_len
            ):
                self.sent = True

        req = types.SimpleNamespace(
            rid="req",
            bootstrap_room=123,
            kv_committed_len=2,
            req_pool_idx=1,
        )
        receiver = Receiver()
        entry = {
            "decode_req": types.SimpleNamespace(req=req, kv_receiver=receiver),
            "metadata_index": -1,
        }
        queue = types.SimpleNamespace(
            _pre_alloc=lambda req, prefix_len, total_prefix_len: torch.tensor(
                [7, 8], dtype=torch.int32
            ),
            kv_manager=types.SimpleNamespace(
                kv_args=types.SimpleNamespace(state_types=[])
            ),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.transfer_backend = TransferBackend.FAKE
        scheduler.server_args = types.SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False
        )
        scheduler.disagg_decode_prealloc_queue = queue
        scheduler.req_to_metadata_buffer_idx_allocator = Allocator()
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )

        Scheduler._pd_flip_target_prealloc_and_send_metadata(scheduler, entry)

        self.assertTrue(receiver.sent)
        self.assertEqual(
            scheduler.disagg_metadata_buffers.bootstrap_room[0, 0].item(), 123
        )

    def test_fake_target_success_releases_placeholder_request(self):
        class Receiver:
            def __init__(self):
                self.has_metadata = False
                self.cleared = False

            def poll(self):
                return KVPoll.Success if self.has_metadata else KVPoll.WaitingForInput

            def clear(self):
                self.cleared = True

            def abort(self):
                raise AssertionError("abort should not be called")

        scheduler = Scheduler.__new__(Scheduler)
        released = []
        freed_metadata = []

        def prealloc(entry):
            entry["decode_req"].kv_receiver.has_metadata = True

        scheduler._pd_flip_target_init_receiver = lambda decode_req: True
        scheduler._pd_flip_target_prealloc_and_send_metadata = prealloc
        scheduler._pd_flip_target_metadata_ready = lambda entry: True
        scheduler._pd_flip_release_target_request = lambda entry: released.append(
            entry["decode_req"].req.rid
        )
        scheduler._pd_flip_free_target_metadata = lambda entry: freed_metadata.append(
            entry["decode_req"].req.rid
        )

        receiver = Receiver()
        req = types.SimpleNamespace(rid="req")
        session = {
            "manifests": [{"rid": "req"}],
            "target_entries": {
                "req": {
                    "decode_req": types.SimpleNamespace(
                        req=req, kv_receiver=receiver
                    ),
                    "phase": "new",
                }
            },
        }

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(released, ["req"])
        self.assertEqual(freed_metadata, ["req"])
        self.assertTrue(receiver.cleared)
        self.assertEqual(session["state"], "target_transferred")


if __name__ == "__main__":
    unittest.main()
