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

    def test_pd_flip_target_hicache_stitch_sends_only_uncached_suffix(self):
        class Allocator:
            def alloc(self):
                return 0

        class Receiver:
            def __init__(self):
                self.sent = None

            def send_metadata(
                self, page_indices, metadata_index, state_indices, decode_prefix_len
            ):
                self.sent = {
                    "page_indices": page_indices.tolist(),
                    "metadata_index": metadata_index,
                    "state_indices": state_indices,
                    "decode_prefix_len": decode_prefix_len,
                }

        class Queue:
            def __init__(self):
                self.prefix_match = types.SimpleNamespace(
                    prefix_indices=torch.tensor([10, 20], dtype=torch.int64),
                    l1_prefix_len=2,
                    decode_prefix_len=3,
                    restore_token_count=1,
                )
                self.prealloc_args = None
                self.prefetch_args = None
                self.kv_manager = types.SimpleNamespace(
                    kv_args=types.SimpleNamespace(state_types=[])
                )

            def _match_prefix_and_lock(self, req):
                return self.prefix_match

            def _pre_alloc(
                self, req, prefix_indices=None, prefix_len=None, total_prefix_len=None
            ):
                self.prealloc_args = {
                    "prefix_indices": prefix_indices.tolist(),
                    "prefix_len": prefix_len,
                    "total_prefix_len": total_prefix_len,
                }
                return torch.tensor([40, 50, 60], dtype=torch.int64)

            def _start_hicache_prefetch(self, req, prefix_match):
                self.prefetch_args = (req.rid, prefix_match.decode_prefix_len)

        req = types.SimpleNamespace(
            rid="req",
            bootstrap_room=123,
            kv_committed_len=6,
            req_pool_idx=1,
            origin_input_ids=[1, 2, 3, 4, 5, 6],
        )
        receiver = Receiver()
        decode_req = types.SimpleNamespace(req=req, kv_receiver=receiver)
        entry = {"decode_req": decode_req, "metadata_index": -1}
        queue = Queue()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.transfer_backend = TransferBackend.FAKE
        scheduler.server_args = types.SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True,
            enable_pd_flip_hicache_stitch=True,
        )
        scheduler.disagg_decode_prealloc_queue = queue
        scheduler.enable_decode_hicache = True
        scheduler.req_to_metadata_buffer_idx_allocator = Allocator()
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)
        scheduler.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [10, 20, 30, 40, 50, 60],
                ],
                dtype=torch.int64,
            )
        )
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )

        Scheduler._pd_flip_target_prealloc_and_send_metadata(scheduler, entry)

        self.assertEqual(queue.prealloc_args["prefix_indices"], [10, 20])
        self.assertEqual(queue.prealloc_args["prefix_len"], 2)
        self.assertEqual(queue.prealloc_args["total_prefix_len"], 3)
        self.assertEqual(queue.prefetch_args, ("req", 3))
        self.assertEqual(decode_req.prefix_match, queue.prefix_match)
        self.assertEqual(receiver.sent["decode_prefix_len"], 3)
        self.assertEqual(receiver.sent["page_indices"], [40, 50, 60])
        self.assertEqual(entry["target_hicache_prefix_len"], 3)
        self.assertEqual(entry["mooncake_hit_len"], 3)
        self.assertEqual(entry["target_prompt_len"], 6)
        self.assertEqual(entry["target_committed_len"], 6)
        self.assertEqual(entry["target_received_suffix_start"], 3)
        self.assertEqual(entry["target_received_suffix_end"], 6)

    def test_pd_flip_target_waits_for_hicache_restore_before_success(self):
        class Receiver:
            def __init__(self):
                self.cleared = False

            def poll(self):
                return KVPoll.Success

            def clear(self):
                self.cleared = True

            def abort(self):
                raise AssertionError("abort should not be called")

        class TransferQueue:
            def __init__(self):
                self.processed = []

            def _process_hicache_local_restores(self, decode_reqs):
                self.processed.extend(decode_reqs)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_decode_hicache = True
        scheduler.disagg_decode_transfer_queue = TransferQueue()
        scheduler._pd_flip_target_metadata_ready = lambda entry: True
        scheduler._pd_flip_release_target_request = lambda entry: self.fail(
            "target must not release before HiCache restore is ready"
        )
        scheduler._pd_flip_free_target_metadata = lambda entry: self.fail(
            "metadata must stay allocated while HiCache restore is pending"
        )

        receiver = Receiver()
        req = types.SimpleNamespace(rid="req")
        decode_req = types.SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            prefix_match=types.SimpleNamespace(needs_local_restore=True),
            hicache_restore_status=types.SimpleNamespace(value="pending"),
        )
        session = {
            "manifests": [{"rid": "req"}],
            "target_entries": {
                "req": {
                    "decode_req": decode_req,
                    "phase": "transferring",
                }
            },
        }

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(scheduler.disagg_decode_transfer_queue.processed, [decode_req])
        self.assertFalse(receiver.cleared)
        self.assertEqual(session["transferred_reqs"], 0)
        self.assertEqual(session["pending_reqs"], 1)

    def test_pd_flip_target_commits_hicache_restore_before_release(self):
        class Receiver:
            def __init__(self):
                self.cleared = False

            def poll(self):
                return KVPoll.Success

            def clear(self):
                self.cleared = True

            def abort(self):
                raise AssertionError("abort should not be called")

        events = []

        class TransferQueue:
            def _process_hicache_local_restores(self, decode_reqs):
                events.append("process_restore")

            def _commit_hicache_local_restore_to_req(self, decode_req):
                events.append("commit_restore")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_decode_hicache = True
        scheduler.disagg_decode_transfer_queue = TransferQueue()
        scheduler._pd_flip_target_metadata_ready = lambda entry: True
        scheduler._pd_flip_release_target_request = lambda entry: events.append(
            "release"
        )
        scheduler._pd_flip_free_target_metadata = lambda entry: events.append(
            "free_metadata"
        )

        receiver = Receiver()
        req = types.SimpleNamespace(rid="req")
        decode_req = types.SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            prefix_match=types.SimpleNamespace(needs_local_restore=True),
            hicache_restore_status=types.SimpleNamespace(value="ready"),
        )
        session = {
            "manifests": [{"rid": "req"}],
            "target_entries": {
                "req": {
                    "decode_req": decode_req,
                    "phase": "transferring",
                }
            },
        }

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(
            events,
            ["process_restore", "commit_restore", "release", "free_metadata"],
        )
        self.assertTrue(receiver.cleared)
        self.assertEqual(session["transferred_reqs"], 1)

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
