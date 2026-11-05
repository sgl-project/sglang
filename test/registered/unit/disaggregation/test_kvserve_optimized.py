"""CPU ownership/order gates. CUDA and transport are explicit test stand-ins."""

import __future__

import ast
import contextlib
import importlib.util
import itertools
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SG_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = SG_ROOT / "python/sglang/srt/disaggregation/mooncake"
spec = importlib.util.spec_from_file_location(
    "_kvserve_composed_helpers", MODULE_ROOT / "kvserve_pipeline.py"
)
helpers = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: helpers}):
    spec.loader.exec_module(helpers)
RestorePipeline = helpers.RestorePipeline


def adapter_classes(fake_torch):
    """Load the actual method ASTs without importing SGLang's GPU entrypoints.

    This also permits the local Python 3.9 CPU gate to check the Python>=3.10
    target sources. Serving uses ordinary package imports; no AST is used there.
    """
    failed = types.SimpleNamespace(Failed="Failed", Success="Success")
    namespace = dict(
        torch=fake_torch,
        msgspec=msgspec,
        threading=threading,
        time=time,
        hashlib=__import__("hashlib"),
        KVPoll=failed,
        StateType=types.SimpleNamespace(MAMBA="MAMBA"),
        READY=b"KVSERVE_READY_V3",
        ACK=b"KVSERVE_ACK_V3",
        ReceiveCredits=None,
        logger=logging.getLogger(__name__),
        endpoint=lambda host, port: f"tcp://{host}:{port}",
    )
    transport_spec = importlib.util.spec_from_file_location(
        "_kvserve_composed_transport", MODULE_ROOT / "kvserve_transport.py"
    )
    transport = importlib.util.module_from_spec(transport_spec)
    transport_spec.loader.exec_module(transport)
    namespace["ReceiveCredits"] = transport.ReceiveCredits
    for filename, name in (
        ("kvserve.py", "KVServeAdapter"),
        ("kvserve_optimized.py", "OptimizedKVServeAdapter"),
    ):
        tree = ast.parse((MODULE_ROOT / filename).read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
        )
        if name == "OptimizedKVServeAdapter":
            namespace.update(
                BoundedSourcePool=helpers.BoundedSourcePool,
                EncodedTransferPipeline=helpers.EncodedTransferPipeline,
                InlineNotifications=helpers.InlineNotifications,
                RestorePipeline=helpers.RestorePipeline,
            )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(MODULE_ROOT / filename),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
    return namespace["KVServeAdapter"], namespace["OptimizedKVServeAdapter"], namespace


RealAdapter, _, _ = adapter_classes(
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(device=lambda _: contextlib.nullcontext())
    )
)


class Event:
    def __init__(self):
        self.done = threading.Event()

    def record(self, stream):
        self.done = stream.done

    def synchronize(self):
        if not self.done.wait(3):
            raise RuntimeError("Fake GPU completion timed out")

    def elapsed_time(self, other):
        return 0.25


class Adapter(RealAdapter):
    def __init__(self, count=2):
        self.lock = threading.Lock()
        self.slot_locks = [threading.Lock() for _ in range(count)]
        self.device, self.poisoned, self.verify = "fake", False, False
        self.chunk_size, self.cache_blocks, self.page_size = 1024, 100000, 1
        self.expected_rooms = {1: {10, 11, 12, 13}}
        self.restored_rooms = {1: {0: set(), 1: set()}}
        self.acks, self.records, self.failures = queue.Queue(), queue.Queue(), []
        self.submitted = queue.Queue()
        self.raise_decode = False
        self.block_record = None
        streams = []
        for _ in range(count):
            done = threading.Event()

            def sync(done=done):
                if not done.wait(3):
                    raise RuntimeError("Fake GPU stream timed out")

            streams.append(types.SimpleNamespace(done=done, synchronize=sync))
        self.runtime = types.SimpleNamespace(
            slots=[
                types.SimpleNamespace(arena_a=types.SimpleNamespace(numel=lambda: 1024))
                for _ in range(count)
            ],
            streams=streams,
            wire_profile="profile",
            aux_bytes=lambda *a: 32,
            decode=self.decode,
        )
        self.manager = types.SimpleNamespace(
            request_status={1: True},
            check_status=lambda room: "Working",
            record_failure=lambda *args: self.failures.append(args),
            update_status=lambda *args: None,
        )

    def decode(self, slot, groups, part, *args, **kwargs):
        self.submitted.put((slot, part))
        if self.raise_decode:
            raise RuntimeError("Fake submission failed after enqueue")
        event = Event()
        event.record(self.runtime.streams[slot])
        return event

    def token_ids(self, pages, page_size, tokens):
        if len(pages) != tokens:
            raise ValueError("bad token/page count")
        return pages

    def reply(self, meta, result):
        self.acks.put((meta.copy(), result.copy()))

    def record(self, record):
        if self.block_record:
            self.block_record.wait(3)
        self.records.put(record)

    def control_failed(self, error):
        self.poisoned = True
        self.failures.append(error)


def meta(slot=0, part=0, pages=None, seq="1", room=1):
    pages = [10] if pages is None else pages
    return dict(
        slot_id=slot,
        room=room,
        epoch="epoch",
        sequence=seq,
        version=3,
        profile="profile",
        indices=pages,
        tokens=len(pages),
        part=part,
        aux_bytes=32,
        codec_bytes=128,
        u8_bytes=256,
        reply_host="peer",
        reply_port=1,
    )


class TestReceiveGate(CustomTestCase):
    def pipeline(self, expect_unsettled=False):
        adapter = Adapter()
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device=lambda _: contextlib.nullcontext())
        )
        fake_module = types.SimpleNamespace(
            KVPoll=types.SimpleNamespace(Failed="Failed")
        )
        pipeline = RestorePipeline(adapter, fake_torch, fake_module.KVPoll.Failed)

        def close():
            try:
                pipeline.close()
            except RuntimeError as error:
                if not (expect_unsettled and "Unsettled" in str(error)):
                    raise

        def release_waiters():
            for stream in adapter.runtime.streams:
                stream.done.set()
            if adapter.block_record:
                adapter.block_record.set()

        self.addCleanup(close)
        self.addCleanup(release_waiters)
        return adapter, pipeline

    def test_slots_submit_without_wait_and_ack_only_after_own_gpu(self):
        a, p = self.pipeline()
        p.accept([meta(), meta(1, 1, seq="2")])
        self.assertEqual(
            {a.submitted.get(timeout=1), a.submitted.get(timeout=1)}, {(0, 0), (1, 1)}
        )
        self.assertTrue(a.acks.empty())
        a.runtime.streams[1].done.set()
        _, ack = a.acks.get(timeout=1)
        self.assertEqual(ack["sequence"], "2")
        self.assertIsNone(ack["restore_gpu_ms"])
        self.assertEqual(a.restored_rooms[1][1], {10})
        self.assertEqual(a.restored_rooms[1][0], set())
        self.assertTrue(a.acks.empty())
        a.runtime.streams[0].done.set()
        self.assertEqual(a.acks.get(timeout=1)[1]["sequence"], "1")

    def test_invalid_frames_fail_before_gpu_submission(self):
        cases = (
            ({"profile": "wrong"}, "receive session"),
            ({"part": 2}, "destination pages"),
            ({"indices": [10, 10], "tokens": 2}, "destination pages"),
            ({"aux_bytes": 31}, "auxiliary length"),
            ({"codec_bytes": 1025}, "encoded length"),
        )
        for updates, message in cases:
            with self.subTest(updates=updates):
                adapter = Adapter()
                torch = types.SimpleNamespace(
                    cuda=types.SimpleNamespace(
                        device=lambda _: contextlib.nullcontext()
                    )
                )
                pipeline = RestorePipeline(adapter, torch, "Failed")
                try:
                    pipeline.accept([dict(meta(), **updates)])
                    result = adapter.acks.get(timeout=1)[1]
                finally:
                    pipeline.close()
                self.assertIn(message, result["error"])
                self.assertTrue(adapter.submitted.empty())

    def test_bounded_duplicate_slot_rejected_atomically(self):
        a, p = self.pipeline()
        with self.assertRaises(ValueError):
            p.accept([meta(), meta(seq="2")])
        self.assertEqual(p.busy, {})
        p.accept([meta()])
        a.submitted.get(timeout=1)
        with self.assertRaises(ValueError):
            p.accept([meta(seq="3")])
        self.assertEqual(len(p.busy), 1)
        a.runtime.streams[0].done.set()
        a.acks.get(timeout=1)

    def test_duplicate_pages_reserved_across_slots_and_error_ack(self):
        a, p = self.pipeline()
        p.accept([meta(), meta(1, 0, seq="2")])
        a.submitted.get(timeout=1)
        _, error = a.acks.get(timeout=1)
        self.assertIn("duplicate", error["error"])
        self.assertTrue(a.poisoned)
        a.runtime.streams[0].done.set()
        a.acks.get(timeout=1)

    def test_submission_error_drains_gpu_before_error_ack(self):
        a, p = self.pipeline()
        a.raise_decode = True
        p.accept([meta()])
        a.submitted.get(timeout=1)
        self.assertTrue(a.acks.empty())
        a.runtime.streams[0].done.set()
        self.assertIn("submission failed", a.acks.get(timeout=1)[1]["error"])
        self.assertTrue(a.poisoned)

    def test_cancel_never_submits_gpu_and_returns_cancel_ack(self):
        a, p = self.pipeline()
        a.manager.request_status.clear()
        p.accept([meta()])
        self.assertTrue(a.acks.get(timeout=1)[1]["cancelled"])
        self.assertTrue(a.submitted.empty())

    def test_room_drop_wait_and_ack_precedes_file_record(self):
        a, p = self.pipeline()
        a.block_record = threading.Event()
        p.accept([meta()])
        a.submitted.get(timeout=1)
        dropped = threading.Event()
        drop = threading.Thread(target=lambda: (p.wait_room(1), dropped.set()))
        drop.start()
        self.assertFalse(dropped.wait(0.01))
        a.runtime.streams[0].done.set()
        a.acks.get(timeout=1)
        self.assertTrue(dropped.wait(1))
        self.assertTrue(a.records.empty())
        a.block_record.set()
        drop.join()

    def test_shutdown_drains_before_close_and_rejects_new_ready(self):
        a, p = self.pipeline()
        p.accept([meta()])
        a.submitted.get(timeout=1)
        closed = threading.Event()
        thread = threading.Thread(target=lambda: (p.close(), closed.set()))
        thread.start()
        self.assertFalse(closed.wait(0.01))
        a.runtime.streams[0].done.set()
        self.assertTrue(closed.wait(1))
        with self.assertRaises(RuntimeError):
            p.accept([meta()])
        thread.join()

    def test_cancel_is_nonblocking_and_wait_failure_keeps_room_quarantined(self):
        a, p = self.pipeline()
        p.accept([meta()])
        a.submitted.get(timeout=1)
        started = time.monotonic()
        p.cancel_room(1)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(p.room_is_idle(1))
        a.runtime.streams[0].done.set()
        self.assertTrue(a.acks.get(timeout=1)[1]["cancelled"])
        self.assertTrue(p.room_is_idle(1))
        self.assertEqual(a.restored_rooms[1][0], set())

    def test_failed_gpu_wait_is_not_reported_as_idle(self):
        a, p = self.pipeline(expect_unsettled=True)

        def fail_decode(slot, groups, part, *args, **kwargs):
            a.submitted.put((slot, part))

            def fail():
                raise RuntimeError("unsettled CUDA wait")

            return types.SimpleNamespace(synchronize=fail)

        a.runtime.decode = fail_decode
        p.accept([meta()])
        a.submitted.get(timeout=1)
        self.assertIn("unsettled", a.acks.get(timeout=1)[1]["error"])
        self.assertFalse(p.room_is_idle(1))
        with self.assertRaisesRegex(RuntimeError, "Unsettled"):
            p.close()
        with self.assertRaisesRegex(RuntimeError, "Unsettled"):
            p.close()


class TestSourceAndLeaseGate(CustomTestCase):
    def test_source_pool_is_bounded_and_not_part_pinned(self):
        source = helpers.BoundedSourcePool(2)
        seen, lock = [], threading.Lock()
        started, release = threading.Barrier(3), threading.Event()
        self.addCleanup(source.shutdown)
        self.addCleanup(release.set)

        def work(slot, part):
            with lock:
                seen.append((slot, part))
            started.wait(2)
            release.wait(2)

        jobs = [source.submit(work, 0) for _ in range(2)]
        started.wait(2)
        self.assertEqual({slot for slot, _ in seen}, {0, 1})
        release.set()
        for future in jobs:
            future.result(2)
        source.shutdown()
        self.assertEqual(source.queue.maxsize, 2)
        with self.assertRaises(RuntimeError):
            source.submit(work, 1)
        source.shutdown()  # base adapter repeats shutdown safely.

    def test_backpressure_and_close_do_not_hold_worker_state_lock(self):
        source = helpers.BoundedSourcePool(1)
        release, entered, third = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.addCleanup(source.shutdown)
        self.addCleanup(release.set)
        first = source.submit(lambda _: (entered.set(), release.wait(2)))
        self.assertTrue(entered.wait(1))
        second = source.submit(lambda _: None)
        thread = threading.Thread(
            target=lambda: (source.submit(lambda _: None), third.set())
        )
        thread.start()
        self.assertFalse(third.wait(0.02))
        self.assertLessEqual(source.queue.qsize(), 1)
        release.set()
        self.assertTrue(third.wait(1))
        first.result(1)
        second.result(1)
        thread.join(1)
        source.shutdown()

    def test_output_bank_lives_until_real_write_and_failure_quarantines(self):
        started, release = threading.Event(), threading.Event()

        def write(lease, job):
            started.set()
            release.wait(2)
            return 3

        tx = helpers.EncodedTransferPipeline(
            [(object(), object())], write, lambda _: None, workers=1
        )
        lease = tx.acquire()
        future = tx.submit(lease, {})
        self.assertTrue(started.wait(1))
        self.assertEqual(tx.available, [])
        release.set()
        self.assertEqual(future.result(1), 3)
        self.assertIs(tx.acquire(), lease)
        tx.discard_unsubmitted(lease)
        tx.close()
        tx.close()
        errors = []

        def bad(*_):
            raise RuntimeError("unsettled RDMA")

        tx = helpers.EncodedTransferPipeline(
            [(object(), object())], bad, errors.append, workers=1
        )
        lease = tx.acquire()
        with self.assertRaisesRegex(RuntimeError, "unsettled"):
            tx.submit(lease, {}).result(1)
        self.assertEqual(lease.state, "quarantined")
        self.assertEqual(tx.available, [])
        self.assertTrue(errors)
        tx.close()
        tx.close()


class TestConstructionGate(CustomTestCase):
    def test_manager_selects_the_same_pipeline_even_with_retired_flag(self):
        tree = ast.parse((MODULE_ROOT / "conn.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        node.body = [
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name == "set_kv_buffer_tensors"
        ]
        node.bases = []
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(MODULE_ROOT / "conn.py"),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        manager_type = namespace["MooncakeKVManager"]
        calls = []
        mod = types.ModuleType("sglang.srt.disaggregation.mooncake.kvserve_optimized")
        mod.OptimizedKVServeAdapter = lambda *args: calls.append(args) or "adapter"
        with (
            patch.dict(os.environ, {"SGLANG_KVSERVE_OPTIMIZED": "0"}),
            patch.dict(sys.modules, {mod.__name__: mod}),
        ):
            native = object.__new__(manager_type)
            native.kvserve_enabled = False
            native.set_kv_buffer_tensors(["k"], ["v"], 1, [0])
            self.assertEqual(calls, [])
            compressed = object.__new__(manager_type)
            compressed.kvserve_enabled = True
            compressed.set_kv_buffer_tensors(["k"], ["v"], 1, [0])
        self.assertEqual(compressed.kvserve, "adapter")
        self.assertEqual(calls, [(compressed, ["k"], ["v"], 1, [0])])

    def test_role_specific_construction_and_close(self):
        @dataclass
        class Config:
            enabled: bool = True
            uses_aggregate: bool = False
            slot_count: int = 2
            logical_blocks_per_chunk: int = 2

            @classmethod
            def from_extra_config(cls, _):
                return cls()

        class Tensor:
            ndim = 4
            shape = (32, 1, 1, 2)
            dtype = "bf16"

            def numel(self):
                return 32

            def data_ptr(self):
                return id(self)

        class Runtime:
            def __init__(self, actions):
                self.actions, self.config = actions, Config()
                self.plan = types.SimpleNamespace(
                    reserved_bytes=128, retained_driver_bytes=8
                )
                self.slots = [
                    types.SimpleNamespace(arena_a=Tensor(), aux=Tensor())
                    for _ in range(2)
                ]
                self.output_slots = [(Tensor(), Tensor()) for _ in range(4)]
                self.stream = types.SimpleNamespace(synchronize=lambda: None)
                self.bulk, self.wire_profile = Tensor(), "profile"
                row = types.SimpleNamespace(num_heads=1, head_size=2, sections=[])
                self.layout = types.SimpleNamespace(parts=lambda _: [row])

            def bind_kv_caches(self, *_):
                pass

            def registered_memory(self):
                return [123], [128]

            def shutdown(self):
                self.actions.append("shutdown")

        fake_torch = types.SimpleNamespace(
            device=lambda *_: "fake",
            cuda=types.SimpleNamespace(
                Event=lambda **_: object(), device=lambda _: contextlib.nullcontext()
            ),
        )
        _, optimized, namespace = adapter_classes(fake_torch)
        namespace.update(
            os=os, json=json, Path=Path, uuid=uuid, itertools=itertools, asdict=asdict
        )
        config_module = types.ModuleType("kvserve_v1.compression.runtime")
        config_module.CacheSpec = lambda **kwargs: types.SimpleNamespace(**kwargs)
        config_module.CompressionConfig = Config
        runtime_module = types.ModuleType("sglang.srt.disaggregation.kvserve_runtime")
        runtime_module.parse_compression_config = lambda _: Config()

        with tempfile.NamedTemporaryFile(mode="w") as config_file:
            config_file.write("{}")
            config_file.flush()
            with (
                patch.dict(os.environ, {"SGLANG_KVSERVE_CONFIG": config_file.name}),
                patch.dict(
                    sys.modules,
                    {
                        config_module.__name__: config_module,
                        runtime_module.__name__: runtime_module,
                    },
                ),
            ):
                for role in ("prefill", "decode"):
                    with self.subTest(role=role):
                        actions = []
                        runtime = Runtime(actions)
                        runtime_module.take_runtime = lambda _: runtime
                        engine = types.SimpleNamespace(
                            batch_register=lambda *_: 0,
                            batch_deregister=lambda _: actions.append("deregister"),
                        )
                        manager = types.SimpleNamespace(
                            kv_args=types.SimpleNamespace(
                                gpu_id=0, state_types=[], num_draft_entries=0
                            ),
                            server_args=types.SimpleNamespace(
                                enable_pd_role_switch=False
                            ),
                            is_mla_backend=False,
                            is_hybrid_mla_backend=False,
                            attn_tp_size=1,
                            attn_cp_size=1,
                            dcp_size=1,
                            pp_size=1,
                            enable_staging=False,
                            enable_deferred_decode_kv_release=True,
                            engine=engine,
                            disaggregation_mode=role,
                            _worker_threads=[],
                            _send_multipart_locked=lambda *_: None,
                        )
                        adapter = optimized(manager, [Tensor()], [Tensor()], 1, [0])
                        try:
                            self.assertIsInstance(
                                adapter.notifier, helpers.InlineNotifications
                            )
                            self.assertEqual(
                                adapter._source_pool is not None, role == "prefill"
                            )
                            self.assertEqual(
                                adapter._receive_pipeline_v2 is not None,
                                role == "decode",
                            )
                        finally:
                            adapter.close()
                        self.assertEqual(actions, ["deregister", "shutdown"])


class TestComposedAdapterGate(CustomTestCase):
    def fixture(
        self,
        deferred_gpu=False,
        automatic_ack=True,
        encode_error=False,
        completion_error=False,
        stream_error=False,
    ):
        pointers, waits, records, metas, actions = {}, [], [], [], []
        gpu = threading.Event()
        if not deferred_gpu:
            gpu.set()

        class Tensor:
            def __init__(self, size):
                self.data = bytearray(size)
                pointers[id(self)] = self

            def data_ptr(self):
                return id(self)

            def numel(self):
                return len(self.data)

            def __getitem__(self, sl):
                value = Tensor(0)
                value.data = self.data[sl]
                return value

            def cpu(self):
                return self

            def numpy(self):
                return self

            def tobytes(self):
                return bytes(self.data)

        class Completion:
            def synchronize(self):
                waits.append("complete")
                if completion_error:
                    raise RuntimeError("completion wait failed")
                if not gpu.wait(3):
                    raise RuntimeError("GPU fixture timed out")

        class Stream:
            def synchronize(self):
                waits.append("stream")
                if stream_error:
                    raise RuntimeError("stream wait failed")
                if not gpu.wait(3):
                    raise RuntimeError("GPU fixture timed out")

        class Encoded:
            event = Completion()
            u8_bytes = 8
            raw_tail_bytes = 0

            @property
            def codec_bytes(self):
                if not gpu.is_set():
                    raise AssertionError("length read before completion")
                return 4

        class Runtime:
            def __init__(self):
                self.slots = [
                    types.SimpleNamespace(arena_a=Tensor(16), aux=Tensor(4))
                    for _ in range(2)
                ]
                self.streams = [Stream() for _ in self.slots]
                self.output_slots = [(Tensor(16), Tensor(4)) for _ in range(4)]
                self.wire_profile = "default"
                self.layout = types.SimpleNamespace(
                    parts=lambda part: [types.SimpleNamespace(num_heads=1, head_size=2)]
                )
                self.calls = []

            def aux_bytes(self, *_):
                return 2

            def encode_into(self, slot, groups, part, **kwargs):
                if encode_error:
                    raise RuntimeError("encode enqueue failed")
                out, aux = kwargs["output"], kwargs["auxiliary"]
                self.calls.append((slot, part, out))
                out.data[:4] = bytes([part, groups[0][0], 1, 2])
                aux.data[:2] = b"AX"
                return Encoded()

            def shutdown(self):
                actions.append("shutdown")

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device=lambda _: contextlib.nullcontext())
        )
        base, optimized, namespace = adapter_classes(fake_torch)
        runtime = Runtime()
        engine = types.SimpleNamespace(
            batch_deregister=lambda ptrs: actions.append("deregister")
        )
        a = object.__new__(optimized)

        def acknowledge(meta):
            a.handle_ack(
                [
                    namespace["ACK"],
                    msgspec.msgpack.encode(
                        [
                            dict(
                                epoch=meta["epoch"],
                                sequence=meta["sequence"],
                                restore_ms=0.2,
                                restore_gpu_ms=None,
                            )
                        ]
                    ),
                ]
            )

        def notify(address, frames):
            self.assertEqual(frames[0], namespace["READY"])
            values = msgspec.msgpack.decode(frames[1])
            metas.extend(values)
            if automatic_ack:
                for meta in values:
                    acknowledge(meta)

        def transfer(session, blocks):
            self.assertTrue(gpu.is_set())
            self.assertEqual([b[2] for b in blocks], [4, 2])
            return 0

        manager = types.SimpleNamespace(
            disaggregation_mode="prefill",
            engine=engine,
            _send_multipart_locked=notify,
            _transfer_data=transfer,
            _worker_threads=[],
            _poison_unsettled_send=lambda room: actions.append("poison"),
            local_ip="self",
            rank_port=2,
            record_failure=lambda *x: None,
            update_status=lambda *x: None,
        )

        def initialize(self, *_):
            self.manager = manager
            self.runtime = runtime
            self.device = "fake"
            self.verify = True
            self.has_mamba = False
            self.page_size = 1
            self.chunk_size = 2
            self.cache_blocks = 100
            self.epoch = "epoch"
            self.poisoned = False
            self.pending = {}
            self.credit_pools = {}
            self.sequence = itertools.count()
            self.pending_lock = threading.Lock()
            self.submit_lock = threading.Lock()
            self.lock = threading.Lock()
            self.expected_rooms = {}
            self.restored_rooms = {}
            self.registered_ptrs = [1]
            self.record = records.append
            self.notifier = helpers.InlineNotifications(
                manager._send_multipart_locked, self.control_failed, self.record
            )
            self._source_unsettled_rooms = set()

        with patch.object(base, "__init__", initialize):
            optimized.__init__(a, manager, [], [], 1, [])
        peer = dict(
            version=3,
            has_mamba=False,
            profile="default",
            page_size=1,
            slots=[dict(ptrs=[90 + i, 190 + i], capacities=[16, 4]) for i in range(8)],
        )
        request = types.SimpleNamespace(
            mooncake_session_id="peer", endpoint="peer", dst_port=1
        )
        return (
            a,
            runtime,
            request,
            types.SimpleNamespace(kvserve=peer),
            gpu,
            waits,
            records,
            metas,
            actions,
            acknowledge,
        )

    def test_ready_wire_metadata_digest_and_credit_ack(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            automatic_ack=False
        )
        _, pool = a._prepare_send(request, peer, [0], [3], 1)
        frame = a.send_frame(0, 7, request, peer.kvserve, [0], [3], 0, pool).result(2)
        self.assertFalse(frame["event"].is_set())
        self.assertEqual(len(pool.available), 7)
        self.assertEqual(
            metas[0],
            dict(
                version=3,
                epoch="epoch",
                slot_id=0,
                profile="default",
                room=7,
                sequence="0",
                part=0,
                indices=[3],
                tokens=1,
                codec_bytes=4,
                u8_bytes=8,
                aux_bytes=2,
                reply_host="self",
                reply_port=2,
                digest=__import__("hashlib")
                .sha256(bytes([0, 0, 1, 2]) + b"AX")
                .hexdigest(),
            ),
        )
        ack(metas[0])
        self.assertTrue(frame["event"].is_set())
        self.assertEqual(len(pool.available), 8)
        a.finish_frame(frame)
        self.assertEqual(a.pending, {})
        a.close()

    def test_real_send_drains_ack_and_reuses_source_before_destination_credit(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            automatic_ack=False
        )
        done = threading.Event()
        errors = []

        def send():
            try:
                a.send(7, request, peer, list(range(8)), list(range(8)), 8)
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=send)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while len(metas) < 8 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(len(metas), 8)
            self.assertFalse(done.is_set())
            self.assertEqual(len(r.calls), 8)
            self.assertEqual(len(a.credit_pools["peer"].available), 0)
            self.assertEqual(len(a._tx_v2.available), 4)
        finally:
            for meta in list(metas):
                ack(meta)
            thread.join(2)
            if done.is_set():
                a.close()
        self.assertFalse(errors)
        self.assertEqual(len([x for x in records if x.get("event") == "send"]), 8)
        self.assertEqual(actions, ["deregister", "shutdown"])
        self.assertEqual(waits, ["complete"] * 8)
        self.assertTrue({s for s, _, _ in r.calls} <= {0, 1})

    def test_metadata_and_ready_follow_actual_encode_completion(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            deferred_gpu=True
        )
        completed = threading.Event()
        errors = []

        def send():
            try:
                a.send(7, request, peer, [0], [0], 1)
            except Exception as e:
                errors.append(e)
            finally:
                completed.set()

        thread = threading.Thread(target=send)
        thread.start()
        deadline = time.monotonic() + 1
        while not waits and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertFalse(metas)
        self.assertFalse(completed.is_set())
        gpu.set()
        thread.join(2)
        self.assertTrue(completed.is_set())
        self.assertFalse(errors)
        self.assertEqual(len(metas), 2)
        a.close()

    def test_encode_failure_keeps_registered_memory_alive(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            encode_error=True
        )
        with self.assertRaisesRegex(RuntimeError, "encode enqueue"):
            a.send(7, request, peer, [0], [0], 1)
        with self.assertRaises(RuntimeError):
            a.close()
        with self.assertRaises(RuntimeError):
            a.close()
        self.assertNotIn("deregister", actions)
        self.assertNotIn("shutdown", actions)
        self.assertTrue(any(lease.state == "quarantined" for lease in a._tx_v2.leases))
        self.assertFalse(metas)

    def test_source_pages_stay_quarantined_after_failed_completion_and_drain(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            completion_error=True, stream_error=True
        )
        with self.assertRaisesRegex(RuntimeError, "completion wait"):
            a.send(7, request, peer, [0], [0], 1)
        self.assertFalse(a.source_encode_release_safe(7))
        self.assertTrue(a.source_encode_release_safe(8))
        self.assertIn("stream", waits)
        with self.assertRaises(RuntimeError):
            a.close()
        self.assertNotIn("deregister", actions)
        self.assertNotIn("shutdown", actions)

    def test_successful_error_path_stream_drain_releases_source_pages(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture(
            completion_error=True
        )
        with self.assertRaisesRegex(RuntimeError, "completion wait"):
            a.send(7, request, peer, [0], [0], 1)
        self.assertIn("stream", waits)
        self.assertTrue(a.source_encode_release_safe(7))
        # Source reads drained, but failed output leases remain quarantined.
        with self.assertRaises(RuntimeError):
            a.close()
        self.assertNotIn("deregister", actions)

    def test_invalid_role_rejected_before_base_allocations(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device=lambda _: contextlib.nullcontext())
        )
        base, optimized, _ = adapter_classes(fake_torch)
        called = []
        with patch.object(base, "__init__", lambda *args: called.append(True)):
            with self.assertRaisesRegex(ValueError, "explicit Prefill"):
                optimized(
                    types.SimpleNamespace(disaggregation_mode="other"), [], [], 1, []
                )
        self.assertFalse(called)

    def test_codec_worker_cannot_close_its_own_adapter(self):
        a, r, request, peer, gpu, waits, records, metas, actions, ack = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "lifecycle owner"):
            a._source_pool.submit(lambda slot: a.close()).result(1)
        self.assertFalse(getattr(a, "_closing", False))
        a.close()


if __name__ == "__main__":
    unittest.main()
