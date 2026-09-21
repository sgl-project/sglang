"""CPU-only contract/control-flow checks; no CUDA or SGLang installation needed.

Load the actual production class bodies with fake GPU/transport collaborators.
These tests exercise admission and lifetime decisions, not GPU kernels or RDMA.
"""

from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import importlib.util
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
if os.environ.get("SGLANG_COMPRESSION_STANDALONE_TEST") == "1":
    for name, relative in [
        ("sglang", "python/sglang"),
        ("sglang.srt", "python/sglang/srt"),
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / relative)]
            sys.modules[name] = module
SRT = ROOT / "python/sglang/srt"
PROTOCOL = SRT / "disaggregation/compression/protocol.py"
spec = importlib.util.spec_from_file_location("pd_compression_test_protocol", PROTOCOL)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


def load_body(path, symbol, namespace, owner=None):
    tree = ast.parse((SRT / path).read_text())
    nodes = tree.body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = next(
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), namespace)
    return namespace[symbol]


class TestBackgroundProgress(unittest.TestCase):
    def test_compression_lifetimes_block_destructive_idle_gate(self):
        modes = NS(PREFILL="prefill", DECODE="decode")
        is_idle = load_body(
            "managers/scheduler.py",
            "is_fully_idle",
            dict(DisaggregationMode=modes),
            owner="Scheduler",
        )
        runtime = NS(transport_failed=False, shared=NS(idle=lambda: True))
        manager = NS(compression_mode="lz4", compression_runtime=runtime)
        adapter = NS(is_idle=lambda: False)
        obj = NS(
            running_batch=NS(is_empty=lambda: True),
            chunked_req=None,
            dllm_manager=NS(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            waiting_queue=[],
            grammar_manager=NS(grammar_queue=[]),
            disaggregation_mode=modes.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_prefill_bootstrap_queue=NS(queue=[], kv_manager=manager),
            enable_hisparse=False,
            enable_hierarchical_cache=True,
            tree_cache=NS(
                ongoing_write_through={},
                ongoing_load_back={},
                enable_storage=False,
                background_work_is_idle=lambda: adapter.is_idle(),
            ),
            _engine_paused=False,
            decode_offload_manager=None,
            disagg_decode_prealloc_queue=NS(
                queue=[], retracted_queue=[], kv_manager=manager
            ),
            disagg_decode_transfer_queue=NS(
                queue=[],
                has_pending_deferred_releases=lambda: False,
                staging_handler=NS(is_idle=lambda: True),
            ),
        )
        self.assertFalse(is_idle(obj), "L2 async restore/queued backup is still live")
        self.assertTrue(is_idle(obj, for_health_check=True))
        adapter.is_idle = lambda: True
        self.assertTrue(is_idle(obj))
        obj.disaggregation_mode = modes.DECODE
        obj.enable_hierarchical_cache = False
        transfer = obj.disagg_decode_transfer_queue
        transfer.has_pending_deferred_releases = lambda: True
        self.assertFalse(is_idle(obj), "remote abort drain is still unconfirmed")
        self.assertTrue(is_idle(obj, for_health_check=True))
        transfer.has_pending_deferred_releases = lambda: False
        transfer.staging_handler.is_idle = lambda: False
        self.assertFalse(is_idle(obj), "local restore is active or quarantined")
        transfer.staging_handler.is_idle = lambda: True
        runtime.transport_failed = True
        self.assertFalse(is_idle(obj))
        runtime.transport_failed = False
        runtime.shared.idle = lambda: False
        self.assertFalse(is_idle(obj))
        runtime.shared.idle = lambda: True
        self.assertTrue(is_idle(obj))

    def test_split_preserves_page_identity_and_host_handles(self):
        import torch

        full = "FULL"
        parent_data = NS(
            lock_ref=0,
            session_ref=0,
            session_ids=None,
            metadata={},
            value=None,
            host_value=None,
        )
        child_data = NS(
            lock_ref=3,
            session_ref=2,
            metadata={"compression_page_refs": (91, 92, 93)},
            value=torch.tensor([8, 4, 6]),
            host_value=torch.tensor([101, 102, 103]),
        )
        split = load_body(
            "mem_cache/unified_cache/components/full.py",
            "redistribute_on_node_split",
            {},
            owner="FullComponent",
        )
        split(
            NS(component_type=full),
            NS(key=[10], component_data={full: parent_data}),
            NS(component_data={full: child_data}),
        )
        self.assertEqual(parent_data.metadata["compression_page_refs"], (91,))
        self.assertEqual(child_data.metadata["compression_page_refs"], (92, 93))
        self.assertEqual(
            parent_data.host_value.tolist() + child_data.host_value.tolist(),
            [101, 102, 103],
        )
        self.assertEqual(
            parent_data.value.tolist() + child_data.value.tolist(), [8, 4, 6]
        )

    def test_no_batch_yields_only_for_compression_or_existing_storage(self):
        clock = NS(monotonic=lambda: 0, sleep=Mock())
        on_idle = load_body(
            "managers/scheduler.py",
            "on_idle",
            dict(
                time=clock,
                LOAD_STALL_REFRESH_S=1,
                SCHEDULER_STAGE_IDLE="idle",
                scheduler_stage_method=lambda _: lambda fn: fn,
            ),
            owner="Scheduler",
        )
        pending = False
        obj = NS(
            maybe_send_health_check_signal=Mock(),
            is_fully_idle=lambda: False,
            metrics_reporter=Mock(),
            _last_stall_publish_ts=0,
            tree_cache=NS(has_pending_background_work=lambda: pending),
            enable_hicache_storage=False,
        )
        on_idle(obj)
        clock.sleep.assert_not_called()
        pending = True
        on_idle(obj)
        clock.sleep.assert_called_once_with(0.001)
        clock.sleep.reset_mock()
        pending = False
        obj.enable_hicache_storage = True
        on_idle(obj)
        clock.sleep.assert_called_once_with(0)
        clock.sleep.reset_mock()
        obj.enable_hicache_storage = False
        obj.tree_cache = NS()  # Legacy/external caches without the optional method.
        on_idle(obj)
        clock.sleep.assert_not_called()

    def test_active_batch_event_poll_does_not_sleep(self):
        clock = NS(sleep=Mock())
        poll = load_body(
            "managers/scheduler.py",
            "_process_hicache_events",
            dict(
                time=clock,
                get_memory=lambda: NS(enable_flexkv=False),
                envs=NS(SGLANG_KV_COMPRESSION_TRACE_HANDOFF=NS(get=lambda: False)),
            ),
            owner="Scheduler",
        )
        cache = NS(
            check_hicache_events=Mock(), has_pending_background_work=lambda: True
        )
        poll(
            NS(
                enable_hierarchical_cache=True,
                tree_cache=cache,
                enable_hicache_storage=False,
            )
        )
        cache.check_hicache_events.assert_called_once()
        clock.sleep.assert_not_called()


class TestProtocol(unittest.TestCase):
    def descriptor(self, **kw):
        return protocol.ChunkDescriptor(
            **dict(
                nonce="generation-1", encoding="lz4", raw_bytes=64, wire_bytes=32, **kw
            )
        )

    def test_round_trip(self):
        desc = self.descriptor()
        self.assertEqual(protocol.ChunkDescriptor.from_bytes(desc.to_bytes()), desc)
        desc.validate(nonce="generation-1", raw_bytes=64, capacity=80, mode="lz4")

    def test_bad_descriptors(self):
        for field, value in [
            ("version", 99),
            ("raw_bytes", True),
            ("wire_bytes", -1),
            ("nonce", ""),
            ("encoding", "unknown"),
            ("sha256", "oops"),
        ]:
            with self.subTest(field=field):
                value_map = json.loads(self.descriptor().to_bytes())
                value_map[field] = value
                with self.assertRaises(ValueError):
                    protocol.ChunkDescriptor.from_bytes(json.dumps(value_map).encode())
        with self.assertRaises(ValueError):
            protocol.ChunkDescriptor.from_bytes(b"x" * 1025)

    def test_mismatched_range_capacity_generation_and_mode(self):
        for values in [
            dict(nonce="old", raw_bytes=64, capacity=80, mode="lz4"),
            dict(nonce="generation-1", raw_bytes=63, capacity=80, mode="lz4"),
            dict(nonce="generation-1", raw_bytes=64, capacity=31, mode="lz4"),
            dict(nonce="generation-1", raw_bytes=64, capacity=80, mode="passthrough"),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.descriptor().validate(**values)

    def test_raw_and_incompressible(self):
        raw = protocol.ChunkDescriptor("x", "raw", 64, 64)
        protocol.ChunkDescriptor.from_bytes(raw.to_bytes())
        with self.assertRaises(ValueError):
            protocol.ChunkDescriptor.from_bytes(
                protocol.ChunkDescriptor("x", "raw", 64, 63).to_bytes()
            )
        # Forced compression may expand; it remains valid within reserved space.
        protocol.ChunkDescriptor("x", "lz4", 64, 80).validate(
            nonce="x", raw_bytes=64, capacity=80, mode="lz4"
        )

    def test_peer_agreement(self):
        for mode in protocol.MODES:
            protocol.check_peer(protocol.capability(mode), protocol.capability(mode))
        for remote in ["off", protocol.capability("passthrough"), "pd-kv-v0/lz4"]:
            with self.assertRaises(ValueError):
                protocol.check_peer(protocol.capability("lz4"), remote)
        with self.assertRaises(ValueError):
            protocol.validate_mode("typo")

    def test_forced_test_peers_cannot_mix_with_normal_mode(self):
        with self.assertRaises(ValueError):
            protocol.check_peer(
                protocol.capability("lz4", force=True),
                protocol.capability("lz4", force=False),
            )

    def test_source_lifetime_includes_queued_tasks(self):
        tasks = protocol.RoomTasks()
        tasks.add(1)
        tasks.add(1)
        released = threading.Event()
        waiter = threading.Thread(target=lambda: (tasks.drain(1), released.set()))
        waiter.start()
        self.assertFalse(released.wait(0.02))
        tasks.finish(1)
        self.assertTrue(tasks.pending(1))
        self.assertFalse(released.wait(0.02))
        tasks.finish(1)
        self.assertTrue(released.wait(1))
        waiter.join()
        self.assertFalse(tasks.pending(1))


class FakeStagingHandler:
    def advance_scatter(self, req):
        # Model the existing allocation/event completion gate.
        for event, alloc_id in list(req._chunk_events):
            if event.query():
                req._chunk_events.remove((event, alloc_id))
        receiver = self._room_to_receiver[req.req.bootstrap_room]
        req._staging_scatter_done = (
            req._staging_all_success
            and not req._chunk_events
            and all(x[0] < 0 for x in receiver.chunk_staging_infos)
        )

    def release_room(self, room, req, receiver):
        self.released.append(room)


class TestReceiverLifecycle(unittest.TestCase):
    def setUp(self):
        ns = dict(
            DecodeStagingHandler=FakeStagingHandler,
            time=time,
            ChunkDescriptor=protocol.ChunkDescriptor,
            BufferDrainError=protocol.BufferDrainError,
            KVPoll=NS(Failed=0),
            logger=logging.getLogger(__name__),
        )
        cls = load_body(
            "disaggregation/mooncake/compression.py",
            "CompressedDecodeStagingHandler",
            ns,
        )
        self.handler = cls.__new__(cls)
        h = self.handler
        h._restore_lock = threading.Lock()
        h._restore_tasks = {}
        h._quarantined = {}
        h._restore_executor = Mock()
        self.future = concurrent.futures.Future()
        h._restore_executor.submit.return_value = self.future
        self.req = NS(
            req=NS(bootstrap_room=7),
            _staging_failed=False,
            _staging_all_success=True,
            _staging_scatter_done=False,
            _chunk_events=[],
        )
        self.receiver = NS(
            session_id="peer",
            compression_nonce="new",
            chunk_staging_infos=[(8, 0, 0, 80, 1)],
        )
        h._room_to_decode_req = {7: self.req}
        h._room_to_receiver = {7: self.receiver}
        h._writer_counts = {}
        h.runtime = NS(
            mode="lz4", chunk_tokens=1024, bytes_per_token=64, verify=False, force=False
        )
        h.kv_manager = NS(
            check_status=lambda r: 3,
            record_failure=Mock(),
            update_status=Mock(),
            _staging_ctx=NS(room_receivers={7: self.receiver}, room_bootstrap={7: []}),
        )
        h.released = []

    def send(self, nonce="new", raw=64, wire=32):
        self.handler.handle_compressed_chunk(
            7,
            0,
            0,
            1,
            "peer",
            protocol.ChunkDescriptor(
                nonce, "pages", raw, wire, pages=((0, wire, "lz4"),)
            ).to_bytes(),
        )

    def test_network_done_does_not_admit_until_restore_done(self):
        self.send()
        self.handler.advance_scatter(self.req)
        self.assertFalse(self.req._staging_scatter_done)
        self.future.set_result(NS(query=lambda: True))
        self.handler.advance_scatter(self.req)
        self.assertTrue(self.req._staging_scatter_done)

    def test_pending_gpu_event_still_blocks_admission(self):
        self.send()
        self.future.set_result(NS(query=lambda: False))
        self.handler.advance_scatter(self.req)
        self.assertFalse(self.req._staging_scatter_done)

    def test_duplicate_and_stale_arrivals(self):
        self.send(nonce="old")
        self.handler._restore_executor.submit.assert_not_called()
        self.send()
        self.send()
        self.handler._restore_executor.submit.assert_called_once()

    def test_bad_length_never_submits_restore(self):
        self.send(wire=81)
        self.assertTrue(self.req._staging_failed)
        self.handler._restore_executor.submit.assert_not_called()

    def test_verify_requires_digest_before_restore_submission(self):
        self.handler.runtime.verify = True
        self.send()
        self.assertTrue(self.req._staging_failed)
        self.handler._restore_executor.submit.assert_not_called()

    def test_verified_forced_descriptor_can_submit(self):
        self.handler.runtime.verify = self.handler.runtime.force = True
        self.handler.handle_compressed_chunk(
            7,
            0,
            0,
            1,
            "peer",
            protocol.ChunkDescriptor(
                "new",
                "pages",
                64,
                32,
                sha256="a" * 64,
                pages=((0, 32, "lz4"),),
            ).to_bytes(),
        )
        self.assertFalse(self.req._staging_failed)
        self.handler._restore_executor.submit.assert_called_once()

    def test_forced_receiver_rejects_raw_page_before_restore_submission(self):
        self.handler.runtime.force = True
        self.handler.handle_compressed_chunk(
            7,
            0,
            0,
            1,
            "peer",
            protocol.ChunkDescriptor(
                "new",
                "pages",
                64,
                64,
                sha256="a" * 64,
                pages=((0, 64, "raw"),),
            ).to_bytes(),
        )
        self.assertTrue(self.req._staging_failed)
        self.handler._restore_executor.submit.assert_not_called()

    def test_restore_error_never_admits(self):
        self.send()
        self.future.set_exception(ValueError("bad payload"))
        self.handler.advance_scatter(self.req)
        self.assertTrue(self.req._staging_failed)
        self.assertFalse(self.req._staging_scatter_done)
        self.assertEqual(self.receiver.chunk_staging_infos[0][0], 8)

    def test_cancel_waits_for_active_restore_before_release(self):
        self.send()
        self.future.set_running_or_notify_cancel()
        done = threading.Event()
        worker = threading.Thread(
            target=lambda: (self.handler.unregister_decode_req(7), done.set())
        )
        worker.start()
        self.assertFalse(done.wait(0.02))
        self.assertEqual(self.handler.released, [])
        self.future.set_result(NS(query=lambda: True))
        self.assertTrue(done.wait(1))
        worker.join()
        self.assertEqual(self.handler.released, [7])

    def test_cancel_queued_restore_and_next_request(self):
        self.send()
        self.assertFalse(self.handler.is_idle())
        self.handler.unregister_decode_req(7)
        self.assertTrue(self.future.cancelled())
        self.assertEqual(self.handler.released, [7])
        self.assertEqual(self.handler._restore_tasks, {})
        self.assertTrue(self.handler.is_idle())

    def test_undrained_cuda_never_releases_pages(self):
        self.send()
        self.future.set_exception(protocol.BufferDrainError("CUDA not quiescent"))
        self.handler.advance_scatter(self.req)
        self.assertTrue(self.req._staging_failed)
        with self.assertRaises(protocol.BufferDrainError):
            self.handler.unregister_decode_req(7)
        self.assertEqual(self.handler.released, [])
        self.assertIs(self.handler._quarantined[7], self.req)
        self.assertFalse(self.handler.is_idle())


class TestTransportFailure(unittest.TestCase):
    def test_failed_wire_registration_does_not_allocate_again(self):
        import torch

        cls = load_body(
            "disaggregation/mooncake/compression.py",
            "CompressionRuntime",
            dict(torch=torch, time=time, BufferDrainError=protocol.BufferDrainError),
        )
        runtime = cls.__new__(cls)
        runtime.transport_failed = False
        runtime.local = threading.local()
        runtime.outputs = []
        runtime.device = torch.device("cpu")
        runtime.wire_capacity = 128
        future = concurrent.futures.Future()
        future.set_result(NS(data=torch.zeros(64, dtype=torch.uint8)))
        lease = NS(future=future, source="new", close=Mock())
        runtime.shared = NS(acquire_pages=Mock(return_value=[lease]))
        runtime.manager = NS(
            _register_staging_memory=Mock(
                side_effect=RuntimeError("registration failed")
            )
        )
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                runtime.encode_pages([0], [1], None, Mock())
        self.assertTrue(runtime.transport_failed)
        self.assertEqual(len(runtime.outputs), 1)
        runtime.manager._register_staging_memory.assert_called_once()
        runtime.shared.acquire_pages.assert_called_once()
        lease.close.assert_called_once()

    def test_failed_rdma_blocks_further_writes(self):
        path = "disaggregation/mooncake/conn.py"
        ns = {"logger": logging.getLogger(__name__)}
        transfer = load_body(path, "_transfer_data", ns, owner="MooncakeKVManager")
        quarantine = load_body(
            path, "_quarantine_compression_transport", ns, owner="MooncakeKVManager"
        )
        manager = NS(
            compression_mode="lz4",
            compression_runtime=NS(transport_failed=False),
            compression_uncertain_rooms=set(),
            _compression_active_room=7,
            engine=NS(batch_transfer_sync=Mock(return_value=-1)),
        )
        manager._quarantine_compression_transport = lambda: quarantine(manager)
        self.assertEqual(transfer(manager, "peer", [(100, 200, 32)]), -1)
        self.assertEqual(manager.compression_uncertain_rooms, {7})
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            transfer(manager, "peer", [(100, 200, 32)])
        manager.engine.batch_transfer_sync.assert_called_once()

    def test_uncertain_rdma_cannot_ack_safe_release(self):
        ack = load_body(
            "disaggregation/common/conn.py",
            "_send_abort_ack",
            {},
            owner="CommonKVManager",
        )
        manager = NS(compression_uncertain_rooms={7}, _send_multipart_locked=Mock())
        ack(manager, "peer", 1234, 7)
        manager._send_multipart_locked.assert_not_called()

    def test_registration_failure_is_not_silently_ignored(self):
        register = load_body(
            "disaggregation/mooncake/conn.py",
            "_register_staging_memory",
            {},
            owner="MooncakeKVManager",
        )
        manager = NS(
            compression_mode="lz4", engine=NS(batch_register=Mock(return_value=-1))
        )
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            register(manager, 100, 32)


class TestConfigurationAndRegistration(unittest.TestCase):
    def test_configuration_rejects_unsupported_combinations(self):
        cfg = dict(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            tp_size=1,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            page_size=1,
            attention_backend="flashinfer",
            disable_radix_cache=True,
            enable_hierarchical_cache=False,
            enable_hisparse=False,
            disaggregation_decode_enable_radix_cache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            speculative_algorithm=None,
            chunked_prefill_size=1024,
            hicache_size=8,
            hicache_host_memory_mode="cache",
            hicache_write_policy="write_through",
            hicache_storage_backend=None,
        )
        settings = NS(
            host="off", tree="python", cpp=False, force=False, verify=False, mode="lz4"
        )
        validate = load_body(
            "arg_groups/pd_disaggregation_hook.py",
            "_validate_pd_compression",
            dict(
                resolving_view=lambda args: args,
                os=NS(getenv=lambda key: None),
                envs=NS(
                    SGLANG_HICACHE_KV_COMPRESSION=NS(get=lambda: settings.host),
                    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=NS(
                        get=lambda: settings.tree
                    ),
                    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=NS(get=lambda: settings.cpp),
                    SGLANG_PD_KV_COMPRESSION_FORCE=NS(get=lambda: settings.force),
                    SGLANG_PD_KV_COMPRESSION_VERIFY=NS(get=lambda: settings.verify),
                    SGLANG_PD_KV_COMPRESSION=NS(get=lambda: settings.mode),
                    SGLANG_KV_COMPRESSION_WORKSPACE_MB=NS(get=lambda: 512),
                    SGLANG_RUST_SERVER=NS(get=lambda: False),
                    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NS(get=lambda: None),
                ),
                model_config_of=lambda args: NS(
                    hf_config=NS(architectures=["Qwen3ForCausalLM"])
                ),
            ),
        )
        validate(NS(**cfg))
        for key, value in [
            ("tp_size", 2),
            ("page_size", 16),
            ("enable_hierarchical_cache", True),
            ("disable_radix_cache", False),
            ("disable_overlap_schedule", False),
            ("disaggregation_transfer_backend", "nixl"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(NS(**(cfg | {key: value})))
        settings.force = True
        with self.assertRaises(ValueError):
            validate(NS(**cfg))
        settings.verify = True
        validate(NS(**cfg))
        with self.assertRaises(ValueError):
            validate(NS(**(cfg | {"enable_hierarchical_cache": True})))
        settings.force = False
        settings.host = "lz4"
        composed = cfg | {
            "enable_hierarchical_cache": True,
            "disable_radix_cache": False,
        }
        validate(NS(**composed))
        settings.force = True
        validate(NS(**composed))
        settings.host = "passthrough"
        with self.assertRaisesRegex(ValueError, "FORCE"):
            validate(NS(**composed))
        settings.host = "lz4"
        settings.force = False
        for key, value in [
            ("hicache_size", 0),
            ("hicache_host_memory_mode", "buffer_only"),
            ("hicache_write_policy", "write_back"),
            ("hicache_storage_backend", "file"),
            ("enable_session_radix_cache", True),
            ("enable_unified_cache_external_linker", True),
            ("enable_lmcache", True),
            ("enable_flexkv", True),
            ("radix_cache_backend", "external"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(NS(**(composed | {key: value})))
        settings.tree = "rust"
        with self.assertRaises(ValueError):
            validate(NS(**composed))

    def test_prefill_manager_uses_scheduler_cache(self):
        runtime = object()
        manager = NS(compression_mode="lz4")
        init = load_body(
            "disaggregation/prefill.py",
            "_init_kv_manager",
            dict(
                get_kv_class=lambda backend, kind: (
                    NS if kind == "args" else lambda *a: manager
                ),
                KVClassType=NS(KVARGS="args", MANAGER="manager"),
                DisaggregationMode=NS(PREFILL="prefill"),
                _is_npu=False,
                _transfer_start_layer=lambda **kw: 0,
                build_kv_layer_ids=lambda **kw: [],
                setup_state_kv_args=Mock(),
                DeepSeekV4TokenToKVPool=type("V4Pool", (), {}),
                get_disagg=lambda: NS(disaggregation_ib_device="mlx5_1"),
                envs=NS(SGLANG_DISAGG_STAGING_BUFFER=NS(get=lambda: False)),
            ),
            owner="PrefillBootstrapQueue",
        )
        queue = NS(
            transfer_backend="mooncake",
            tp_rank=0,
            pp_rank=0,
            token_to_kv_pool=NS(
                get_contiguous_buf_infos=lambda: ([], [], []), head_num=8, page_size=1
            ),
            draft_token_to_kv_pool=None,
            is_mla_backend=False,
            metadata_buffers=NS(get_buf_infos=lambda: ([], [], [])),
            scheduler=NS(
                ps=NS(dp_rank=0, gpu_id=0),
                rust_server=None,
                server_args=None,
                tp_worker=NS(model_runner=NS(kv_cache_dtype_str="bf16")),
                model_config=NS(
                    hf_text_config=None,
                    num_hidden_layers=36,
                    get_total_num_kv_heads=lambda: 8,
                ),
                tree_cache=NS(get_kv_compression_context=lambda: (runtime, None)),
            ),
        )
        self.assertIs(init(queue).shared_compression_runtime, runtime)
        queue.scheduler.tree_cache = NS()
        self.assertIs(init(queue), manager)

    def test_launcher_gpu_binding_and_forced_test_guards(self):
        launcher = ROOT / "test/manual/kv_transfer/launch_pd_compression.sh"
        base = dict(
            os.environ,
            ROLE="prefill",
            MODEL_PATH="/model",
            IB_DEVICE="mlx5_1",
            DRY_RUN="1",
            COMPRESSION_MODE="lz4",
            HICACHE_COMPRESSION="off",
            ENABLE_HICACHE="0",
            SGLANG_PD_KV_COMPRESSION_FORCE="0",
            SGLANG_PD_KV_COMPRESSION_VERIFY="0",
        )
        base.pop("CUDA_VISIBLE_DEVICES", None)

        def run(**kw):
            return subprocess.run(
                ["bash", str(launcher)], env=base | kw, capture_output=True, text=True
            )

        for allocation in ["3", "GPU-01234567-89ab-cdef-0123-456789abcdef"]:
            completed = run(NVIDIA_VISIBLE_DEVICES=allocation)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("CUDA_VISIBLE_DEVICES=" + allocation, completed.stdout)
        for allocation in ["3,4", "void", "all"]:
            self.assertNotEqual(run(NVIDIA_VISIBLE_DEVICES=allocation).returncode, 0)
        forced = dict(
            NVIDIA_VISIBLE_DEVICES="3",
            SGLANG_PD_KV_COMPRESSION_FORCE="1",
            SGLANG_PD_KV_COMPRESSION_VERIFY="1",
        )
        self.assertEqual(run(**forced).returncode, 0)
        self.assertEqual(
            run(
                **(forced | dict(ENABLE_HICACHE="1", HICACHE_COMPRESSION="lz4"))
            ).returncode,
            0,
        )
        for role, host, verify in [
            ("decode", "lz4", "1"),
            ("prefill", "passthrough", "1"),
            ("prefill", "lz4", "0"),
        ]:
            with self.subTest(role=role, host=host, verify=verify):
                self.assertNotEqual(
                    run(
                        **(
                            forced
                            | dict(
                                ROLE=role,
                                ENABLE_HICACHE="1",
                                HICACHE_COMPRESSION=host,
                                SGLANG_PD_KV_COMPRESSION_VERIFY=verify,
                            )
                        )
                    ).returncode,
                    0,
                )

    def test_registration_old_and_extended_frames(self):
        # Execute the actual parser on the old wire layout and new trailing fields.
        cls = load_body(
            "disaggregation/mooncake/conn.py",
            "KVArgsRegisterInfo",
            dict(
                __name__=__name__,
                dataclasses=dataclasses,
                struct=struct,
                unpack_int_lists=lambda data, fmt: [],
                StagingRegisterInfo=NS(from_zmq_fields=lambda *a, **k: None),
            ),
        )
        frames = [
            b"-1",
            b"127.0.0.1",
            b"1234",
            b"peer",
            struct.pack("Q", 100),
            struct.pack("Q", 200),
            b"",
            b"0",
            b"1",
            b"64",
            b"",
            b"",
            b"",
            b"",
            struct.pack("Q", 300),
            b"1024",
            b"1",
            b"0",
            b"",
        ]
        old = cls.from_zmq(frames)
        self.assertEqual(old.compression_capability, "off")
        self.assertEqual(old.staging_base_ptr, 300)
        updated = cls.from_zmq(
            frames + [protocol.capability("lz4").encode(), b"layout"]
        )
        self.assertEqual(updated.compression_capability, protocol.capability("lz4"))
        self.assertEqual(updated.compression_layout, "layout")


class TestSenderAdmission(unittest.TestCase):
    def test_failure_and_success_wait_for_source_tasks(self):
        poll = load_body(
            "disaggregation/mooncake/conn.py",
            "poll",
            {"KVPoll": NS(Transferring=2, Success=3, Failed=0, Bootstrapping=1)},
            owner="MooncakeKVSender",
        )
        for status in (0, 3):
            tasks = protocol.RoomTasks()
            tasks.add(7)
            sender = NS(
                bootstrap_room=7,
                conclude_state=None,
                trace_ctx=Mock(),
                kv_mgr=NS(
                    compression_mode="lz4",
                    compression_tasks=tasks,
                    check_status=lambda r: status,
                    _staging_outstanding={},
                ),
            )
            self.assertEqual(poll(sender), 2)
            tasks.finish(7)
            self.assertEqual(poll(sender), status)

    def test_uncertain_transport_holds_source_and_aux_slots(self):
        poll = load_body(
            "disaggregation/mooncake/conn.py",
            "poll",
            {"KVPoll": NS(Transferring=2)},
            owner="MooncakeKVSender",
        )
        sender = NS(bootstrap_room=7, kv_mgr=NS(compression_uncertain_rooms={7}))
        self.assertEqual(poll(sender), 2)


if __name__ == "__main__":
    unittest.main()
