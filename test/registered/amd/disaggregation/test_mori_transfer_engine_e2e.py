import os
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.srt.disaggregation.mori.conn import (
    KVArgsRegisterInfo,
    MoriKVManager,
    MoriKVReceiver,
    MoriKVSubmissionError,
    MoriPackedDCPSource,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_amd_ci(est_time=900, suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd")


class TestMoriDCPTransfer(unittest.TestCase):
    @staticmethod
    def _make_generation_manager():
        manager = MoriKVManager.__new__(MoriKVManager)
        manager._deferred_ack_lock = threading.Lock()
        manager._deferred_ack_retry_rooms = set()
        manager._staging_outstanding = {}
        manager._room_lifecycle_lock = threading.Lock()
        manager.transfer_lock = threading.Lock()
        manager._room_generations = {}
        manager._room_owners = {}
        manager._retired_room_status = {}
        manager._retired_room_failures = {}
        manager._aborted_generations = {}
        manager._rooms_pending_clear = {}
        manager._deferred_ack_targets = {}
        manager.transfer_infos = {}
        manager.decode_kv_args_table = {}
        manager.request_status = {}
        manager.req_to_decode_prefix_len = {}
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.requires_strict_deferred_release = True
        manager.enable_deferred_decode_kv_release = True
        return manager

    def test_shared_status_notification_keeps_request_epoch(self):
        manager = self._make_generation_manager()
        manager.attn_tp_rank = 0
        manager.pp_size = 1
        manager.attn_cp_size = 1
        manager.pp_rank = 0
        manager.attn_cp_rank = 0
        manager._send_multipart_locked = MagicMock()
        manager._room_owners[23] = "owner-a"
        manager._room_generations[23] = "request-a"
        manager.request_status[23] = KVPoll.Transferring
        manager.transfer_infos[23] = {
            "decode": SimpleNamespace(
                endpoint="10.0.0.2",
                dst_port=9000,
                is_dummy=False,
            )
        }

        status = manager._conclude_owned_transfer(
            23,
            KVPoll.Success,
            "owner-a",
        )

        self.assertEqual(status, KVPoll.Success)
        sent_frames = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(sent_frames[-1], b"request-a")

    def test_decode_tokenizer_epochs_are_unique_for_reused_request_id(self):
        class FakeSamplingParams:
            def __init__(self, **_kwargs):
                pass

            def normalize(self, _tokenizer):
                pass

            def verify(self, _vocab_size):
                pass

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = {}
        manager.sampling_params_class = FakeSamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=32)
        manager.disaggregation_mode = DisaggregationMode.DECODE
        manager.rid_to_state = {
            "reused-request-id": SimpleNamespace(
                time_stats=SimpleNamespace(set_tokenize_finish_time=lambda: None)
            )
        }
        request = GenerateReqInput(
            rid="reused-request-id",
            text="hello",
            sampling_params={},
            bootstrap_room=23,
        )

        first = manager._create_tokenized_object(request, "hello", [1])
        second = manager._create_tokenized_object(request, "hello", [1])

        self.assertIsNotNone(first.disagg_request_epoch)
        self.assertNotEqual(first.disagg_request_epoch, second.disagg_request_epoch)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        prefill = manager._create_tokenized_object(request, "hello", [1])
        self.assertIsNone(prefill.disagg_request_epoch)

    def test_aborted_generation_does_not_poison_reused_room(self):
        manager = self._make_generation_manager()
        manager.decode_kv_args_table = {"decode": SimpleNamespace(dst_dcp_size=4)}
        manager.resolve_kv_replica_factor = MagicMock()
        manager._send_abort_ack = MagicMock(return_value=True)
        payload = lambda generation: [
            b"23",
            b"10.0.0.2",
            b"9000",
            b"decode",
            np.array([7], dtype=np.int32).tobytes(),
            b"1",
            b"",
            b"1",
            b"",
            generation.encode("ascii"),
        ]

        manager.activate_room_owner(23, "owner-a")
        manager._handle_transfer_message(payload("request-a"))
        manager._handle_abort_message(
            [b"ABORT", b"23", b"10.0.0.2", b"9000", b"request-a"]
        )
        self.assertNotIn(23, manager._room_owners)

        manager._handle_transfer_message(payload("request-a"))
        manager._handle_transfer_message(payload("request-b"))
        manager.activate_room_owner(23, "owner-b")

        self.assertEqual(manager.request_status[23], KVPoll.WaitingForInput)
        self.assertEqual(manager._room_generations[23], "request-b")
        manager._handle_abort_message(
            [b"ABORT", b"23", b"10.0.0.2", b"9000", b"request-a"]
        )
        manager._handle_abort_message([b"ABORT", b"23", b"10.0.0.2", b"9000"])
        self.assertEqual(manager.request_status[23], KVPoll.WaitingForInput)
        self.assertEqual(manager._room_owners[23], "owner-b")

        with patch("sglang.srt.disaggregation.mori.conn.MORI_ABORT_TOMBSTONE_LIMIT", 2):
            for room in (30, 31, 32):
                manager._handle_abort_message(
                    [
                        b"ABORT",
                        str(room).encode("ascii"),
                        b"10.0.0.2",
                        b"9000",
                        f"request-{room}".encode("ascii"),
                    ]
                )
        self.assertEqual(len(manager._aborted_generations), 2)

    def test_dcp1_abort_waits_for_counted_chunk_and_cleans_owner(self):
        manager = self._make_generation_manager()
        manager.decode_kv_args_table = {"decode": SimpleNamespace(dst_dcp_size=1)}
        manager.resolve_kv_replica_factor = MagicMock()
        manager.activate_room_owner(23, "owner-a")
        manager._handle_transfer_message(
            [
                b"23",
                b"10.0.0.2",
                b"9000",
                b"decode",
                np.array([7], dtype=np.int32).tobytes(),
                b"1",
                b"",
                b"1",
                b"",
            ]
        )
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager._num_shards = 1
        manager._dcp_pack_worker_count = 0
        manager._staging_outstanding[23] = 0
        manager._transfer_queues = [MagicMock()]
        manager._send_abort_ack = MagicMock(return_value=True)

        def abort_during_admission(_chunk):
            self.assertEqual(manager._staging_outstanding[23], 1)
            manager._handle_abort_message([b"ABORT", b"23", b"10.0.0.2", b"9000"])
            manager._send_abort_ack.assert_not_called()

        manager._transfer_queues[0].put.side_effect = abort_during_admission
        manager.add_transfer_request(
            23,
            np.array([2], dtype=np.int32),
            slice(0, 1),
            False,
            num_kv_tokens=4,
            room_owner="owner-a",
        )
        self.assertTrue(manager._defer_room_clear_if_outstanding(23, "owner-a"))
        manager._staging_outstanding[23] = 0
        manager._maybe_ack_drained_abort(23)
        manager._clear_room_after_drain(23)

        manager._send_abort_ack.assert_called_once_with("10.0.0.2", 9000, 23, None)
        self.assertNotIn(23, manager._room_owners)
        self.assertNotIn(23, manager.request_status)
        self.assertNotIn((23, "owner-a"), manager._retired_room_status)
        self.assertNotIn((23, "owner-a"), manager._retired_room_failures)

    def test_register_info_parses_dcp_geometry(self):
        kv_descs = [object(), object()]
        payload = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"engine",
            b"kv",
            b"aux",
            b"state",
            b"3",
            b"4",
            b"1",
            b"1024",
            b"",
            b"",
            struct.pack("2Q", 1024, 2048),
            struct.pack("2I", 2, 7),
            b"4",
            b"3",
            b"registration-a",
        ]
        fake_engine_desc = SimpleNamespace(key="engine")

        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
                return_value=fake_engine_desc,
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
                side_effect=[kv_descs, []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_lists",
                return_value=[],
            ),
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_kv_item_lens, [1024, 2048])
        self.assertEqual(info.dst_kv_layer_ids, [2, 7])
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)
        self.assertEqual(info.dcp_registration_id, "registration-a")

    def test_register_info_defaults_for_legacy_peer(self):
        kv_descs = [object(), object()]
        payload = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"engine",
            b"kv",
            b"aux",
            b"state",
            b"0",
            b"1",
            b"0",
            b"256",
        ]

        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
                return_value=SimpleNamespace(key="engine"),
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
                side_effect=[kv_descs, []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_lists",
                return_value=[],
            ),
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_kv_item_lens, [256, 256])
        self.assertEqual(info.dst_kv_layer_ids, [])
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)

    def test_send_kvcache_dcp_uses_token_byte_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["src0", "src1"]
        manager.kv_args = SimpleNamespace(num_draft_entries=0)
        submitted_statuses = iter(("status0", "status1"))

        def submit_plan(*args, status_sink=None):
            result = [next(submitted_statuses)]
            if status_sink is not None:
                status_sink.extend(result)
            return result

        manager._submit_batch_transfer_plan = MagicMock(side_effect=submit_plan)
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["dst0", "dst1"],
            dcp_dst_region_indices=[1, 0],
            dcp_token_item_lens=[8, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([2], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=4,
        )

        statuses = manager.send_kvcache_dcp(peer_info, plan)

        self.assertEqual(statuses, ["status0", "status1"])
        first_plan = manager._submit_batch_transfer_plan.call_args_list[0].args[2]
        self.assertEqual(first_plan.local_offsets, [64, 80])
        self.assertEqual(first_plan.remote_offsets, [224, 232])
        self.assertEqual(first_plan.sizes, [8, 8])
        second_plan = manager._submit_batch_transfer_plan.call_args_list[1].args[2]
        self.assertEqual(second_plan.local_offsets, [128, 160])
        self.assertEqual(second_plan.remote_offsets, [448, 464])
        self.assertEqual(second_plan.sizes, [16, 16])

    def test_send_kvcache_dcp_keeps_draft_rows_replicated_and_unpacked(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["target-src", "draft-src"]
        manager.kv_args = SimpleNamespace(num_draft_entries=1)
        submitted_statuses = iter(("target-status", "draft-status"))

        def submit_plan(*args, status_sink=None):
            result = [next(submitted_statuses)]
            if status_sink is not None:
                status_sink.extend(result)
            return result

        manager._submit_batch_transfer_plan = MagicMock(side_effect=submit_plan)
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["draft-dst", "target-dst"],
            dcp_dst_region_indices=[1, 0],
            dcp_token_item_lens=[8, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([3, 4], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=2,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=4,
        )
        packed_src = MoriPackedDCPSource(
            mem_desc="target-pack",
            layer_offsets=[128],
            token_indices=np.arange(2, dtype=np.int64),
        )

        statuses = manager.send_kvcache_dcp(peer_info, plan, packed_src)

        self.assertEqual(statuses, ["target-status", "draft-status"])
        target_call, draft_call = manager._submit_batch_transfer_plan.call_args_list
        self.assertEqual(target_call.args[:2], ("target-pack", "target-dst"))
        self.assertEqual(target_call.args[2].local_offsets, [128])
        self.assertEqual(target_call.args[2].remote_offsets, [112])
        self.assertEqual(target_call.args[2].sizes, [16])
        self.assertEqual(draft_call.args[:2], ("draft-src", "draft-dst"))
        self.assertEqual(draft_call.args[2].local_offsets, [96])
        self.assertEqual(draft_call.args[2].remote_offsets, [448])
        self.assertEqual(draft_call.args[2].sizes, [64])

    def test_send_kvcache_dcp_submits_draft_when_target_plan_is_empty(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["target-src", "draft-src"]
        manager.kv_args = SimpleNamespace(num_draft_entries=1)

        def submit_plan(*args, status_sink=None):
            result = ["draft-status"]
            if status_sink is not None:
                status_sink.extend(result)
            return result

        manager._submit_batch_transfer_plan = MagicMock(side_effect=submit_plan)
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["target-dst", "draft-dst"],
            dcp_dst_region_indices=[0, 1],
            dcp_token_item_lens=[8, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([3], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=2,
            dcp_size=4,
            dcp_rank=3,
            num_kv_tokens=2,
        )
        self.assertEqual(plan.target_src_token_indices.size, 0)

        statuses = manager.send_kvcache_dcp(peer_info, plan)

        self.assertEqual(statuses, ["draft-status"])
        draft_call = manager._submit_batch_transfer_plan.call_args
        self.assertEqual(draft_call.args[:2], ("draft-src", "draft-dst"))
        self.assertEqual(draft_call.args[2].local_offsets, [96])
        self.assertEqual(draft_call.args[2].remote_offsets, [896])
        self.assertEqual(draft_call.args[2].sizes, [32])

    def test_send_kvcache_dcp_empty_plan_preserves_status_sink(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(num_draft_entries=1)
        manager._submit_batch_transfer_plan = MagicMock()
        plan = build_dcp_token_transfer_plan(
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            physical_page_size=2,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=0,
        )
        statuses = ["existing-status"]

        result = manager.send_kvcache_dcp(SimpleNamespace(), plan, status_sink=statuses)

        self.assertIs(result, statuses)
        self.assertEqual(result, ["existing-status"])
        manager._submit_batch_transfer_plan.assert_not_called()

    def test_send_kvcache_dcp_retains_submitted_parts_on_later_failure(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["target-src", "draft0-src", "draft1-src"]
        manager.kv_args = SimpleNamespace(num_draft_entries=2)
        submitted_statuses = iter(("target-status", "draft0-status"))

        def submit_plan(*args, status_sink=None):
            try:
                result = [next(submitted_statuses)]
            except StopIteration:
                raise RuntimeError("draft submit failed")
            if status_sink is not None:
                status_sink.extend(result)
            return result

        manager._submit_batch_transfer_plan = MagicMock(side_effect=submit_plan)
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["target-dst", "draft0-dst", "draft1-dst"],
            dcp_dst_region_indices=[0, 1, 2],
            dcp_token_item_lens=[8, 16, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([3, 4], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=2,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=4,
        )
        statuses = []

        with self.assertRaisesRegex(RuntimeError, "draft submit failed"):
            manager.send_kvcache_dcp(peer_info, plan, status_sink=statuses)

        self.assertEqual(statuses, ["target-status", "draft0-status"])
        self.assertEqual(manager._submit_batch_transfer_plan.call_count, 3)

    def test_add_remote_peer_resolves_dcp_destination_layers(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.decode_kv_args_table = {}
        manager.engine = SimpleNamespace(register_remote_engine=MagicMock())
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_mem_descs = ["src2", "src7"]
        manager._init_dcp_pack_buffers_once = MagicMock()
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=[2, 7],
            kv_item_lens=[512, 1024],
            page_size=64,
            num_draft_entries=0,
        )
        peer_info = KVArgsRegisterInfo(
            endpoint="10.0.0.2",
            dst_port=23456,
            engine_desc=SimpleNamespace(key="peer"),
            dst_kv_mem_descs=["dst7", "dst2"],
            dst_aux_mem_descs=[],
            dst_state_mem_descs=[],
            gpu_id=3,
            decode_tp_size=4,
            decode_tp_rank=3,
            dst_kv_item_len=1024,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_item_lens=[1024, 512],
            dst_kv_layer_ids=[7, 2],
            dst_dcp_size=4,
            dst_dcp_rank=3,
            supports_dcp_drain=True,
        )

        manager._add_remote_peer(peer_info)

        self.assertTrue(peer_info.requires_dcp_relayout)
        self.assertEqual(peer_info.dcp_dst_region_indices, [1, 0])
        self.assertEqual(peer_info.dcp_token_item_lens, [8, 16])
        manager._init_dcp_pack_buffers_once.assert_called_once_with(4)
        self.assertIs(manager.decode_kv_args_table["peer"], peer_info)
        manager.engine.register_remote_engine.assert_called_once_with(
            peer_info.engine_desc
        )

    def test_add_remote_peer_slices_regular_mla_destination_for_pp(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.decode_kv_args_table = {}
        manager.engine = SimpleNamespace(register_remote_engine=MagicMock())
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_mem_descs = ["src2", "src3"]
        manager._init_dcp_pack_buffers_once = MagicMock()
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=[],
            kv_item_lens=[512, 512],
            page_size=64,
            prefill_start_layer=2,
            mla_compression_ratios=None,
            num_draft_entries=0,
        )
        peer_info = KVArgsRegisterInfo(
            endpoint="10.0.0.2",
            dst_port=23456,
            engine_desc=SimpleNamespace(key="peer"),
            dst_kv_mem_descs=["dst0", "dst1", "dst2", "dst3"],
            dst_aux_mem_descs=[],
            dst_state_mem_descs=[],
            gpu_id=3,
            decode_tp_size=4,
            decode_tp_rank=3,
            dst_kv_item_len=512,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_item_lens=[512, 512, 512, 512],
            dst_dcp_size=4,
            dst_dcp_rank=3,
            supports_dcp_drain=True,
        )

        manager._add_remote_peer(peer_info)

        self.assertEqual(peer_info.dcp_dst_region_indices, [2, 3])
        self.assertEqual(peer_info.dcp_token_item_lens, [8, 8])
        manager._init_dcp_pack_buffers_once.assert_called_once_with(4)

    def test_dcp_peer_without_drain_capability_is_rejected(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.decode_kv_args_table = {}
        peer_info = SimpleNamespace(
            engine_key="legacy-peer",
            dst_dcp_size=2,
            supports_dcp_drain=False,
        )

        with self.assertRaisesRegex(RuntimeError, "drain-ACK support"):
            manager._add_remote_peer(peer_info)

    def test_submit_routes_nonzero_dcp_chunk_with_full_destination_pages(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {23: KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        info = SimpleNamespace(
            engine_key="peer",
            is_dummy=False,
            dst_kv_indices=np.array([7], dtype=np.int32),
            decode_prefix_len=0,
            dst_state_indices=[],
            dst_aux_index=-1,
        )
        peer_info = SimpleNamespace(
            requires_dcp_relayout=True,
            dst_dcp_size=2,
            dst_dcp_rank=0,
        )
        manager.transfer_infos = {23: {"peer": info}}
        manager.decode_kv_args_table = {"peer": peer_info}
        manager.kv_args = SimpleNamespace(page_size=4)
        manager.state_mem_descs = []
        manager.update_status = MagicMock()

        def send_kvcache_dcp(*args, status_sink=None):
            status_sink.append("status")
            return status_sink

        manager.send_kvcache_dcp = MagicMock(side_effect=send_kvcache_dcp)

        statuses, _ = manager._submit_kv_transfer(
            23,
            np.array([3], dtype=np.int32),
            slice(1, 2),
            False,
            num_kv_tokens=4,
        )

        self.assertEqual(statuses, ["status"])
        plan = manager.send_kvcache_dcp.call_args.args[1]
        np.testing.assert_array_equal(plan.target_src_token_indices, [12, 14])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [30, 31])


class TestMoriDCPPackedTransfer(unittest.TestCase):
    def test_pack_workers_are_budgeted_and_fall_back_to_raw(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager._dcp_pack_init_lock = threading.Lock()
        manager._dcp_pack_dcp_size = None
        manager._dcp_pack_worker_count = 0
        manager._dcp_pack_disabled_workers = set()
        manager._dcp_pack_buffers = None
        manager._num_shards = 8
        manager.kv_args = SimpleNamespace(kv_item_lens=[1024])

        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack."
                "dcp_pack_buffer_bytes_for_args",
                return_value=3 * 1024**3,
            ),
            patch.object(
                type(envs.SGLANG_MORI_DCP_PACK_BUFFER_BUDGET_GB),
                "get",
                return_value=8.0,
            ),
        ):
            manager._init_dcp_pack_buffers_once(4)

        self.assertEqual(manager._dcp_pack_worker_count, 2)
        self.assertEqual(manager._dcp_pack_dcp_size, 4)
        self.assertEqual(manager._dcp_pack_buffers, [None] * 8)

        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {23: KVPoll.WaitingForInput}
        manager.transfer_infos = {23: {"decode": SimpleNamespace(engine_key="peer")}}
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(requires_dcp_relayout=True)
        }
        manager.transfer_lock = threading.Lock()
        manager._num_shards = 8
        manager._dcp_pack_worker_count = 2
        manager._staging_outstanding = {23: 0}
        manager._deferred_ack_lock = threading.Lock()
        manager._transfer_queues = [MagicMock() for _ in range(8)]

        manager.add_transfer_request(
            23, np.array([2], dtype=np.int32), slice(0, 1), False, num_kv_tokens=4
        )

        manager._transfer_queues[1].put.assert_called_once()

        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.init_dcp_pack_buffers",
            side_effect=RuntimeError("out of memory"),
        ):
            pack_buffer = manager._get_or_init_dcp_pack_buffer(1)

        self.assertIsNone(pack_buffer)
        self.assertEqual(manager._dcp_pack_disabled_workers, {1})
        self.assertIsNone(manager._get_or_init_dcp_pack_buffer(1))

    def test_pack_dcp_rank_excludes_draft_tail(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x2000, 0x3000],
            num_draft_entries=1,
        )
        manager._dcp_pack_mem_descs = {0x1000: "pack-desc"}
        pack_buffer = SimpleNamespace(
            get_size=lambda: 1024,
            get_ptr=lambda: 0x1000,
        )
        peer_info = SimpleNamespace(
            dst_dcp_rank=0,
            dst_dcp_size=2,
            dcp_token_item_lens=[8, 16],
        )
        src_indices = np.array([1, 3], dtype=np.int64)
        packed_by_rank = {}

        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
            return_value=([0x1080], np.arange(2, dtype=np.int64)),
        ) as pack_mock:
            packed = manager._pack_dcp_rank_once(
                pack_buffer,
                peer_info,
                src_indices,
                packed_by_rank,
            )

        self.assertIsNotNone(packed)
        self.assertEqual(packed.mem_desc, "pack-desc")
        self.assertEqual(packed.layer_offsets, [128])
        pack_mock.assert_called_once()
        self.assertEqual(pack_mock.call_args.kwargs["kv_data_ptrs"], [0x2000])
        self.assertEqual(pack_mock.call_args.kwargs["token_item_lens"], [8])

    def test_packed_source_collapses_to_one_block_per_layer(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["raw0", "raw1"]
        manager.kv_args = SimpleNamespace(num_draft_entries=0)
        submitted_statuses = iter(("status0", "status1"))

        def submit_plan(*args, status_sink=None):
            result = [next(submitted_statuses)]
            if status_sink is not None:
                status_sink.extend(result)
            return result

        manager._submit_batch_transfer_plan = MagicMock(side_effect=submit_plan)
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["dst0", "dst1"],
            dcp_dst_region_indices=[1, 0],
            dcp_token_item_lens=[8, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([2], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=4,
        )
        packed_src = MoriPackedDCPSource(
            mem_desc="pack",
            layer_offsets=[100, 200],
            token_indices=np.arange(2, dtype=np.int64),
        )

        statuses = manager.send_kvcache_dcp(peer_info, plan, packed_src)

        self.assertEqual(statuses, ["status0", "status1"])
        first_call, second_call = manager._submit_batch_transfer_plan.call_args_list
        self.assertEqual(first_call.args[:2], ("pack", "dst1"))
        self.assertEqual(first_call.args[2].local_offsets, [100])
        self.assertEqual(first_call.args[2].remote_offsets, [224])
        self.assertEqual(first_call.args[2].sizes, [16])
        self.assertEqual(second_call.args[:2], ("pack", "dst0"))
        self.assertEqual(second_call.args[2].local_offsets, [200])
        self.assertEqual(second_call.args[2].remote_offsets, [448])
        self.assertEqual(second_call.args[2].sizes, [32])

    def test_partial_submission_drains_before_error_propagates(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {23: KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        manager.kv_mem_descs = ["raw0", "raw1"]
        manager.kv_args = SimpleNamespace(page_size=4)
        manager.state_mem_descs = []
        manager._dcp_pack_buffers = None
        inflight_status = object()
        manager.engine = SimpleNamespace(
            allocate_transfer_uid=MagicMock(side_effect=[1, 2]),
            batch_write=MagicMock(
                side_effect=[[inflight_status], RuntimeError("submit failed")]
            ),
        )
        manager.transfer_infos = {
            23: {
                "peer": SimpleNamespace(
                    engine_key="peer",
                    is_dummy=False,
                    dst_kv_indices=np.array([7], dtype=np.int32),
                    decode_prefix_len=0,
                    dst_state_indices=[],
                    dst_aux_index=-1,
                )
            }
        }
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(
                requires_dcp_relayout=True,
                dst_dcp_size=2,
                dst_dcp_rank=0,
                dst_kv_mem_descs=["dst0", "dst1"],
                dcp_dst_region_indices=[0, 1],
                dcp_token_item_lens=[8, 16],
            )
        }
        manager._wait_transfer_completion = MagicMock(return_value=None)
        chunk = SimpleNamespace(
            room=23,
            room_owner=None,
            wait_event=None,
            prefill_kv_indices=np.array([2], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            num_kv_tokens=4,
        )

        with self.assertRaises(MoriKVSubmissionError):
            manager._process_transfer_chunk(chunk, worker_index=0)

        manager._wait_transfer_completion.assert_called_once_with([inflight_status])

    def test_failed_strict_abort_delivery_is_retried(self):
        receiver = object.__new__(MoriKVReceiver)
        receiver.bootstrap_room = 23
        receiver.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 8000}]
        receiver.abort_notified = False
        receiver.abort_generation = "request-a"
        receiver.metadata_published = True
        receiver._abort_pending_infos = None
        receiver._abort_retry_lock = threading.Lock()
        receiver._abort_retry_stopped = False
        receiver._abort_retry_scheduled = False
        receiver._connection_pool_entries = {}
        receiver.kv_mgr = SimpleNamespace(
            enable_deferred_decode_kv_release=True,
            requires_strict_deferred_release=True,
            local_ip="10.0.0.2",
            rank_port=9000,
            failure_lock=threading.Lock(),
            failure_records={},
            request_status={23: KVPoll.WaitingForInput},
            required_prefill_response_num_table={},
            prefill_response_tracker={},
            record_failure=MagicMock(),
            update_status=MagicMock(),
            register_deferred_abort_room=MagicMock(),
            is_abort_release_safe=MagicMock(return_value=False),
            clear_decode_room_generation=MagicMock(),
        )
        receiver._connect_to_bootstrap_server = MagicMock(
            side_effect=RuntimeError("connection refused")
        )
        receiver._schedule_abort_notification_retry = MagicMock()

        receiver.abort()

        self.assertTrue(receiver.abort_notified)
        self.assertFalse(receiver._abort_retry_stopped)
        receiver.kv_mgr.register_deferred_abort_room.assert_called_once_with(
            23, "request-a"
        )
        self.assertEqual(len(receiver._abort_pending_infos), 1)
        receiver._schedule_abort_notification_retry.assert_called_once_with()

        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )
        receiver._send_abort_notification()
        self.assertEqual(len(receiver._abort_pending_infos), 1)
        self.assertEqual(receiver._schedule_abort_notification_retry.call_count, 2)

        receiver.kv_mgr.is_abort_release_safe.return_value = True
        receiver._send_abort_notification()
        self.assertEqual(receiver._abort_pending_infos, {})
        sent_frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(sent_frames[-1], b"request-a")


class MoriTransferEngineBase(PDDisaggregationServerBase):
    port_delta = 0
    prefill_tp = 1
    decode_tp = 1
    decode_base_gpu_id = 1
    required_gpus = 2

    # Subclasses can override to pick a different model or pass extra args.
    model_default = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    model_env_var = "SGLANG_MORI_E2E_TEST_MODEL"
    extra_prefill_args: list = []
    extra_decode_args: list = []

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
            if torch.cuda.device_count() < cls.required_gpus:
                raise unittest.SkipTest(
                    f"MORI PD check requires >= {cls.required_gpus} visible GPUs."
                )
        except Exception as e:
            raise unittest.SkipTest(f"torch is not available/usable: {e}")

        super().setUpClass()

        cls._old_use_aiter = os.environ.get("SGLANG_USE_AITER")
        os.environ["SGLANG_USE_AITER"] = "1"

        # The shared fixture defaults to Mooncake in CI; pin Mori explicitly here.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mori"]

        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        cls._shift_ports()
        cls.model = try_cached_model(
            os.environ.get(cls.model_env_var, cls.model_default)
        )

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_old_use_aiter", None) is None:
            os.environ.pop("SGLANG_USE_AITER", None)
        else:
            os.environ["SGLANG_USE_AITER"] = cls._old_use_aiter
        super().tearDownClass()

    @classmethod
    def _shift_ports(cls):
        if cls.port_delta == 0:
            return

        cls.lb_port = str(int(cls.lb_port) + cls.port_delta)
        cls.prefill_port = str(int(cls.prefill_port) + cls.port_delta)
        cls.decode_port = str(int(cls.decode_port) + cls.port_delta)
        cls.bootstrap_port = str(int(cls.bootstrap_port) + cls.port_delta)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.prefill_tp),
            "--attention-backend",
            "aiter",
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.decode_tp),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
            "--attention-backend",
            "aiter",
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def _assert_generate_smoke(self):
        resp = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertIn("text", out)
        self.assertIsInstance(out["text"], str)
        self.assertGreater(len(out["text"]), 0)


class TestMoriTransferEngineE2E(MoriTransferEngineBase):
    def test_generate_smoke(self):
        self._assert_generate_smoke()


class TestMoriTransferEngineTPMismatchE2E(MoriTransferEngineBase):
    port_delta = 10
    prefill_tp = 2
    decode_tp = 4
    decode_base_gpu_id = 2
    required_gpus = 6

    def test_generate_smoke_tp_mismatch(self):
        self._assert_generate_smoke()


if __name__ == "__main__":
    unittest.main()
