"""CPU contracts for mixed linear/full-attention KVServe transfers."""

import json
import os
import queue
import sys
import tempfile
import threading
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation import kvserve_runtime
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.mooncake.kvserve import KVServeAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestHybridPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.path.write_text(json.dumps({"enable-compression-pipeline": True}))
        self.model = types.SimpleNamespace(
            get_num_kv_heads=lambda _: 2,
            head_dim=256,
            v_head_dim=256,
            num_hidden_layers=16,
        )
        self.kvc = types.SimpleNamespace(
            is_draft_worker=False,
            use_mla_backend=False,
            is_hybrid_swa=False,
            mambaish_config=types.SimpleNamespace(
                full_attention_layer_ids=[1, 3, 6, 10, 11, 14]
            ),
            pp_size=1,
            page_size=1,
            kv_cache_dtype=torch.bfloat16,
            gpu_id=0,
            model_config=self.model,
            layer_info=types.SimpleNamespace(start_layer=2, end_layer=15),
        )
        self.config = types.SimpleNamespace(
            logical_blocks_per_chunk=1024, pipeline=("quantizer",)
        )
        self.context = types.ModuleType("sglang.srt.runtime_context")
        self.context.get_parallel = lambda: types.SimpleNamespace(
            attn_tp_size=1, attn_dcp_size=1
        )
        self.context.get_memory = lambda: types.SimpleNamespace(
            enable_unified_memory=False
        )
        self.context.get_disagg = lambda: types.SimpleNamespace(
            disaggregation_mode="prefill", disaggregation_transfer_backend="mooncake"
        )
        self.api = types.ModuleType("kvserve_v1.compression.runtime")
        self.api.CacheSpec = lambda *args: args
        self.plan = types.SimpleNamespace(reserved_bytes=99, retained_driver_bytes=1)
        self.api.build_memory_plan = Mock(return_value=self.plan)
        self.fused = types.ModuleType("kvserve_v1.compression.runtime.fused_runtime")
        self.fused.create_runtime = Mock(
            return_value=types.SimpleNamespace(
                plan=self.plan, bulk=types.SimpleNamespace(numel=lambda: 64)
            )
        )

    def run_prepare(self):
        modules = {
            self.context.__name__: self.context,
            self.api.__name__: self.api,
            self.fused.__name__: self.fused,
        }
        with (
            patch.dict(
                os.environ, {"SGLANG_KVSERVE_CONFIG": str(self.path)}, clear=True
            ),
            patch.dict(sys.modules, modules),
            patch.dict(kvserve_runtime._runtimes, {}, clear=True),
            patch.object(
                kvserve_runtime, "parse_compression_config", return_value=self.config
            ),
            patch.object(
                kvserve_runtime, "select_execution_backend", return_value="auto"
            ),
        ):
            kvserve_runtime.prepare_runtime(self.kvc)

    def test_sparse_global_layers_and_true_model_depth_for_both_roles(self):
        for role, execution in (("prefill", "encode"), ("decode", "decode")):
            with self.subTest(role=role):
                self.api.build_memory_plan.reset_mock()
                self.context.get_disagg = lambda: types.SimpleNamespace(
                    disaggregation_mode=role, disaggregation_transfer_backend="mooncake"
                )
                self.run_prepare()
                args, kwargs = self.api.build_memory_plan.call_args
                self.assertEqual(
                    list(args[1]),
                    ["layer.3", "layer.6", "layer.10", "layer.11", "layer.14"],
                )
                self.assertEqual(
                    [spec[0] for spec in args[1].values()], [3, 6, 10, 11, 14]
                )
                self.assertEqual(kwargs["num_layers"], 16)
                self.assertEqual(
                    self.fused.create_runtime.call_args.kwargs["execution"], execution
                )

    def test_empty_full_layer_set_fails_before_plan(self):
        self.kvc.mambaish_config.full_attention_layer_ids = []
        with self.assertRaisesRegex(ValueError, "full.attention"):
            self.run_prepare()
        self.api.build_memory_plan.assert_not_called()

    def test_hybrid_unified_memory_fails_before_plan(self):
        self.context.get_memory = lambda: types.SimpleNamespace(
            enable_unified_memory=True
        )
        with self.assertRaisesRegex(ValueError, "unified memory"):
            self.run_prepare()
        self.api.build_memory_plan.assert_not_called()


class TestHybridAdapter(unittest.TestCase):
    def test_peer_mamba_capability_must_match_strictly(self):
        slot = types.SimpleNamespace(
            arena_a=types.SimpleNamespace(data_ptr=lambda: 4096, numel=lambda: 4096),
            aux=types.SimpleNamespace(data_ptr=lambda: 8192, numel=lambda: 128),
        )
        adapter = object.__new__(KVServeAdapter)
        adapter.runtime = types.SimpleNamespace(wire_profile="same-kv", slots=[slot])
        adapter.page_size = 1
        for local in (False, True):
            with self.subTest(local=local):
                adapter.has_mamba = local
                peer = adapter.registration()
                self.assertIs(peer["has_mamba"], local)
                adapter.validate_peer(peer)
                mismatches = [
                    {**peer, "has_mamba": not local},
                    {key: val for key, val in peer.items() if key != "has_mamba"},
                    *({**peer, "has_mamba": value} for value in (None, 0, 1, "false")),
                ]
                for mismatch in mismatches:
                    with self.assertRaises(ValueError):
                        adapter.validate_peer(mismatch)

    def test_mamba_passes_capability_gate_but_swa_and_dsa_fail(self):
        runtime = types.ModuleType("kvserve_v1.compression.runtime")
        runtime.CacheSpec = Mock()
        manager = types.SimpleNamespace(
            kv_args=types.SimpleNamespace(
                gpu_id=0, state_types=[], num_draft_entries=0
            ),
            server_args=types.SimpleNamespace(enable_pd_role_switch=False),
            is_mla_backend=False,
            is_hybrid_mla_backend=False,
            attn_tp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            pp_size=1,
            enable_staging=False,
            enable_deferred_decode_kv_release=False,
        )
        with patch.dict(sys.modules, {runtime.__name__: runtime}):
            for state, error in (
                (StateType.MAMBA, "DEFERRED_DECODE"),
                (StateType.SWA, "non-Mamba state"),
                (StateType.DSA, "non-Mamba state"),
            ):
                with self.subTest(state=state):
                    manager.kv_args.state_types = [state]
                    with self.assertRaisesRegex(ValueError, error):
                        KVServeAdapter(manager, [], [], 1, [])
        runtime.CacheSpec.assert_not_called()


class TestHybridTransfer(unittest.TestCase):
    def setup_transfer(self, source=[[7]], dest=[[5]], state_rc=0, aux_rc=0, send=None):
        manager = object.__new__(MooncakeKVManager)
        events = []
        room = 1
        manager.enable_trace = False
        manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = True
        manager.kvserve_enabled = True
        manager.kv_args = types.SimpleNamespace(
            state_types=[StateType.MAMBA], kv_data_ptrs=[1]
        )
        manager.request_status = {room: KVPoll.Transferring}
        manager.check_status = lambda r: manager.request_status[r]
        manager._staging_outstanding = defaultdict(int)
        manager._deferred_ack_targets = {room: {("127.0.0.1", 1): object()}}
        manager._deferred_ack_poisoned_rooms = set()
        manager._send_abort_ack = Mock(
            side_effect=lambda *_: events.append("abort_ack")
        )
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.failed_session_probes = {}
        manager.state_layout_rejections = {}
        manager.req_to_decode_prefix_len = {}
        req = types.SimpleNamespace(
            room=room,
            is_dummy=False,
            mooncake_session_id="peer",
            dst_kv_indices=np.array([42]),
            dst_device_kv_indices=None,
            dst_state_indices=dest,
            endpoint="127.0.0.1",
            dst_port=1234,
            required_dst_info_num=1,
        )
        manager.transfer_infos = {room: {"peer": req}}
        manager.decode_kv_args_table = {
            "peer": types.SimpleNamespace(
                dst_aux_ptrs=[1], dst_aux_item_lens=[1], requires_dcp_relayout=False
            )
        }
        manager.validate_aux_buffers = Mock()
        manager._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
        manager.kvserve = types.SimpleNamespace(
            send=Mock(side_effect=send or (lambda *_: events.append("kv_ack") or 0))
        )
        manager.maybe_send_extra = Mock(
            side_effect=lambda *_: events.append("state") or state_rc
        )
        manager.send_aux = Mock(side_effect=lambda *_: events.append("aux") or aux_rc)
        manager.record_failure = Mock()

        def failure(**kwargs):
            events.append("failure")
            manager.request_status[room] = KVPoll.Failed

        def conclude(**kwargs):
            events.append(
                "success" if kwargs["status"] == KVPoll.Success else "failure"
            )
            manager.request_status[room] = kwargs["status"]

        manager.conclude_failure = Mock(side_effect=failure)
        manager.conclude_transfer = Mock(side_effect=conclude)
        gate = types.SimpleNamespace(
            release=Mock(side_effect=lambda: events.append("release"))
        )
        chunk = types.SimpleNamespace(
            room=room,
            prefill_kv_indices=np.array([3]),
            index_slice=slice(0, 1),
            is_last_chunk=True,
            prefill_aux_index=0,
            state_indices=source,
            num_kv_tokens=1,
            staging_counted=False,
            wait_event=None,
            indices_event=None,
            _kvserve_source_gate=gate,
        )
        work = queue.Queue()
        work.put(chunk)
        work.put(None)
        return manager, events, gate, work

    def test_state_and_aux_wait_for_compressed_restore_ack_before_success(self):
        started, allow = threading.Event(), threading.Event()
        events = []

        def send(*_):
            events.append("kv_start")
            started.set()
            if not allow.wait(3):
                raise RuntimeError("test did not unblock codec ACK")
            events.append("kv_ack")
            return 0

        manager, worker_events, gate, work = self.setup_transfer(send=send)
        thread = threading.Thread(
            target=manager.transfer_worker, args=(work, None), daemon=True
        )
        thread.start()
        self.assertTrue(started.wait(3))
        self.assertEqual(events, ["kv_start"])
        self.assertEqual(worker_events, [])
        gate.release.assert_not_called()
        allow.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(events, ["kv_start", "kv_ack"])
        self.assertEqual(worker_events[:3], ["state", "aux", "success"])
        gate.release.assert_called_once()
        self.assertNotIn(1, manager._deferred_ack_poisoned_rooms)

    def test_state_or_aux_write_failure_poison_ack_and_keep_source_gate(self):
        for state_rc, aux_rc in ((-1, 0), (0, -1)):
            with self.subTest(state_rc=state_rc, aux_rc=aux_rc):
                manager, events, gate, work = self.setup_transfer(
                    state_rc=state_rc, aux_rc=aux_rc
                )
                manager.transfer_worker(work, None)
                self.assertIn("kv_ack", events)
                self.assertIn("failure", events)
                self.assertNotIn("success", events)
                self.assertNotIn("abort_ack", events)
                self.assertIn(1, manager._deferred_ack_poisoned_rooms)
                gate.release.assert_not_called()
                if state_rc:
                    manager.send_aux.assert_not_called()

    def test_missing_or_invalid_mamba_slots_fail_before_any_write(self):
        cases = (
            (None, [[5]]),
            ([], [[5]]),
            ([None], [[5]]),
            ([[]], [[5]]),
            ([[-1]], [[5]]),
            ([[7, 8]], [[5]]),
            ([[7]], None),
            ([[7]], []),
            ([[7]], [None]),
            ([[7]], [[]]),
            ([[7]], [[-1]]),
            ([[7]], [[5, 6]]),
        )
        for source, dest in cases:
            with self.subTest(source=source, dest=dest):
                manager, events, gate, work = self.setup_transfer(
                    source=source, dest=dest
                )
                manager.transfer_worker(work, None)
                self.assertIn("failure", events)
                self.assertNotIn("success", events)
                manager.kvserve.send.assert_not_called()
                manager.maybe_send_extra.assert_not_called()
                manager.send_aux.assert_not_called()
                self.assertNotIn(1, manager._deferred_ack_poisoned_rooms)
                gate.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
