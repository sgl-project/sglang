# SPDX-License-Identifier: Apache-2.0
"""Small, checkpoint-free CUDA IPC tests in independent spawned processes.

Fatal/abandoned-client tests intentionally leave counted sends outstanding;
the bounded generation policy retains those until the producer exits. No test
repairs counters. CPU manifest/mapping tests live in test_weight_cache_common.py.
"""

import gc
import json
import multiprocessing as mp
import os
import signal
import statistics
import struct
import time
import traceback
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.weight_cache_common import transport
from sglang.weight_cache_common.transport import (
    CudaIpcExporter,
    CudaIpcImporter,
    ExportBudgetExceeded,
)
from sglang.weight_cache_common.traversal import snapshot_module

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-small")


def _model(device):
    module = nn.Module()
    module.child = nn.Module()
    base = torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6)
    module.weight = nn.Parameter(base[1:3, 1:5], requires_grad=False)
    module.weight.input_dim = 1
    module.child.tied = module.weight
    buffer = base[:, ::2]
    module.register_buffer("persistent", buffer)
    module.child.register_buffer("scratch", buffer, persistent=False)
    module.register_buffer("view", base.t(), persistent=False)
    module.register_buffer("raw_bytes", base.view(torch.uint8), persistent=False)
    module.register_buffer("scalar", torch.tensor(3.0, device=device))
    module.register_buffer("empty", torch.empty(0, device=device))
    module.eval()
    module.child.train()
    return module


def _check(module):
    reference = _model("cpu")
    assert module.weight is module.child.tied
    assert module.persistent is module.child.scratch
    assert (
        module.weight.untyped_storage()._cdata == module.view.untyped_storage()._cdata
    )
    assert module.weight.storage_offset() == 7
    assert module.weight.stride() == (6, 1)
    assert module.weight.input_dim == 1
    assert not module.training and module.child.training
    assert set(module.state_dict()) == set(reference.state_dict())
    for name, tensor in snapshot_module(module).tensors.items():
        torch.testing.assert_close(
            tensor.cpu(), snapshot_module(reference).tensors[name]
        )
    with torch.inference_mode():
        result = module.weight @ torch.ones(4, 1, device=module.weight.device)
        torch.testing.assert_close(result.cpu(), reference.weight @ torch.ones(4, 1))


def _consumer(conn, action):
    try:
        generation, manifest, deliveries = conn.recv()
        importer = CudaIpcImporter(generation, manifest)
        if action == "cycle":
            conn.send(("guarded", None))
            while True:
                delivery = conn.recv()
                if delivery is None:
                    break
                module = _model("meta")
                importer.receive(delivery, module, request_id=delivery.request_id)
                _check(module)
                del module
                gc.collect()
                conn.send(("cycle_released", None))
            importer.close()
            conn.send(("released", None))
            return
        if action == "before_import":
            conn.send(("guarded", None))
            conn.recv()  # producer-death test: watcher must kill us here
            raise AssertionError("Lost producer was not detected before import")
        if action == "partial":
            original = transport._open_storage
            calls = 0

            def fail_second(handle, device):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected partial import")
                return original(handle, device)

            module = _model("meta")
            retained_error = None
            with patch.object(transport, "_open_storage", side_effect=fail_second):
                try:
                    importer.receive(
                        deliveries[0], module, request_id=deliveries[0].request_id
                    )
                except RuntimeError as error:
                    assert "injected partial import" in str(error)
                    retained_error = error
                else:
                    raise AssertionError("Partial import did not fail")
            # Even an exception traceback can outlive a partially imported
            # storage. It must not let the caller stop the lifetime guard.
            assert retained_error is not None
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, "Release all imported storage"
            ):
                importer.close()
            del retained_error
            assert all(
                t.device.type == "meta"
                for t in snapshot_module(module).tensors.values()
            )
            del module
            gc.collect()
            importer.close()
            conn.send(("partial", None))
            return

        modules = []
        before = torch.cuda.memory_allocated()
        for delivery in deliveries:
            module = _model("meta")
            bad_generation = replace(generation, nonce=uuid.uuid4().hex)
            with unittest.TestCase().assertRaisesRegex(
                ValueError, "generation/request"
            ):
                importer.receive(
                    replace(delivery, generation=bad_generation),
                    module,
                    request_id=delivery.request_id,
                )
            importer.receive(delivery, module, request_id=delivery.request_id)
            with unittest.TestCase().assertRaisesRegex(ValueError, "already consumed"):
                importer.receive(
                    delivery, _model("meta"), request_id=delivery.request_id
                )
            modules.append(module)
        del module
        torch.cuda.synchronize()
        import_delta = torch.cuda.memory_allocated() - before
        for module in modules:
            _check(module)
        del module
        if len(modules) == 2:
            assert modules[0].weight.data_ptr() == modules[1].weight.data_ptr()
            assert modules[0].weight is not modules[1].weight
        torch.cuda.synchronize()
        gc.collect()
        conn.send(
            (
                "mapped",
                {
                    "allocated_delta": import_delta,
                    "after_forward_delta": torch.cuda.memory_allocated() - before,
                },
            )
        )
        conn.recv()
        # A detached view can outlive its module. Closing the guard must still
        # fail until the underlying C++ storage has no remaining owners.
        retained_view = modules[0].weight.detach()
        modules.clear()
        gc.collect()
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "Release all imported storage"
        ):
            importer.close()
        del retained_view
        gc.collect()
        importer.close()
        conn.send(("released", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        raise
    finally:
        conn.close()


def _producer(conn):
    try:
        exporter = CudaIpcExporter(_model("cuda:0"))
        delivery = exporter.export(uuid.uuid4().hex, generation=exporter.generation)
        conn.send((exporter.generation, exporter.manifest, [delivery]))
        conn.recv()
    finally:
        conn.close()


def _counter(handle):
    # Review-only Linux diagnostic for RefcountedMapAllocator's 64-byte prefix.
    # Each test verifies the initial count is 1 before relying on this layout.
    path = Path("/dev/shm") / os.fsdecode(handle.counter_handle).lstrip("/")
    with path.open("rb") as stream:
        stream.seek(64 + handle.counter_offset * 8)
        return struct.unpack("<q", stream.read(8))[0]


def _resources():
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {
        "rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
        "fds": len(list(Path("/proc/self/fd").iterdir())),
        "shm_files": len(list(Path("/dev/shm").glob(f"torch_{os.getpid()}_*"))),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestWeightCacheCommonIpc(unittest.TestCase):
    def setUp(self):
        self.context = mp.get_context("spawn")
        self.children = []
        self.connections = []

    def tearDown(self):
        for child in self.children:
            if child.is_alive():
                child.kill()
            child.join(5)
        for connection in self.connections:
            connection.close()
        gc.collect()
        torch.cuda.ipc_collect()

    def _start(self, payload, action="normal"):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_consumer, args=(child, action))
        process.start()
        child.close()
        self.children.append(process)
        self.connections.append(parent)
        parent.send(payload)
        return process, parent

    def _message(self, connection, expected):
        self.assertTrue(connection.poll(45), f"Timed out waiting for {expected}")
        status, payload = connection.recv()
        self.assertEqual(status, expected, payload)
        return payload

    def _release(self, process, connection):
        connection.send("release")
        self._message(connection, "released")
        process.join(15)
        self.assertEqual(process.exitcode, 0)

    def _delivery(self, exporter):
        delivery = exporter.export(uuid.uuid4().hex, generation=exporter.generation)
        for handle in delivery.storages:
            if handle.nbytes:
                self.assertEqual(_counter(handle), 1)
        return delivery

    def test_sequential_deliveries_balance_and_do_not_replay_counters(self):
        exporter = CudaIpcExporter(_model("cuda:0"))
        seen = set()
        for _ in range(3):
            delivery = self._delivery(exporter)
            tokens = {
                (h.counter_handle, h.counter_offset)
                for h in delivery.storages
                if h.nbytes
            }
            self.assertTrue(seen.isdisjoint(tokens))
            seen.update(tokens)
            process, connection = self._start(
                (exporter.generation, exporter.manifest, [delivery])
            )
            stats = self._message(connection, "mapped")
            self.assertEqual(stats["allocated_delta"], 0)
            self._release(process, connection)
            for handle in delivery.storages:
                if handle.nbytes:
                    self.assertEqual(_counter(handle), 0)
        self.assertEqual(exporter.stats()["deliveries_reserved"], 3)

    def test_overlapping_clients_have_independent_releases(self):
        module = _model("cuda:0")
        exporter = CudaIpcExporter(module)
        first, second = self._delivery(exporter), self._delivery(exporter)
        a, a_conn = self._start((exporter.generation, exporter.manifest, [first]))
        b, b_conn = self._start((exporter.generation, exporter.manifest, [second]))
        self._message(a_conn, "mapped")
        self._message(b_conn, "mapped")
        self._release(a, a_conn)
        for handle in first.storages:
            if handle.nbytes:
                self.assertEqual(_counter(handle), 0)
        for handle in second.storages:
            if handle.nbytes:
                self.assertEqual(_counter(handle), 1)
        self._release(b, b_conn)
        _check(module)  # concurrent inference did not mutate producer state

    def test_same_client_multiple_deliveries_share_mapping_not_send_reference(self):
        exporter = CudaIpcExporter(_model("cuda:0"))
        deliveries = [self._delivery(exporter), self._delivery(exporter)]
        process, connection = self._start(
            (exporter.generation, exporter.manifest, deliveries)
        )
        self._message(connection, "mapped")
        self._release(process, connection)
        for delivery in deliveries:
            for handle in delivery.storages:
                if handle.nbytes:
                    self.assertEqual(_counter(handle), 0)

    def test_fatal_client_retention_is_bounded_not_refunded(self):
        exporter = CudaIpcExporter(_model("cuda:0"), max_deliveries=2)
        first = self._delivery(exporter)
        process, connection = self._start(
            (exporter.generation, exporter.manifest, [first])
        )
        self._message(connection, "mapped")
        process.kill()
        process.join(10)
        self.assertEqual(process.exitcode, -signal.SIGKILL)
        for handle in first.storages:
            if handle.nbytes:
                self.assertEqual(_counter(handle), 1)
        second = self._delivery(exporter)
        process, connection = self._start(
            (exporter.generation, exporter.manifest, [second])
        )
        self._message(connection, "mapped")
        self._release(process, connection)
        with self.assertRaises(ExportBudgetExceeded):
            exporter.export(uuid.uuid4().hex, generation=exporter.generation)

    def test_repeated_attach_resource_accounting(self):
        count = 16
        exporter = CudaIpcExporter(_model("cuda:0"), max_deliveries=count)
        process, connection = self._start(
            (exporter.generation, exporter.manifest, []), "cycle"
        )
        self._message(connection, "guarded")
        before = _resources()
        export_ms = []
        for _ in range(count):
            start = time.monotonic()
            delivery = exporter.export(uuid.uuid4().hex, generation=exporter.generation)
            export_ms.append((time.monotonic() - start) * 1000)
            for handle in delivery.storages:
                if handle.nbytes:
                    self.assertEqual(_counter(handle), 1)
            connection.send(delivery)
            self._message(connection, "cycle_released")
            for handle in delivery.storages:
                if handle.nbytes:
                    self.assertEqual(_counter(handle), 0)
        after = _resources()
        delta = {key: after[key] - before[key] for key in before}
        print(
            "IPC resource sample: "
            + json.dumps(
                {
                    "cycles": count,
                    "delta": delta,
                    "export_ms_median": statistics.median(export_ms),
                    "budget": exporter.stats(),
                },
                sort_keys=True,
            )
        )
        self.assertEqual(delta["cuda_allocated_bytes"], 0)
        self.assertLess(delta["rss_bytes"], 64 * 1024 * 1024)
        self.assertLessEqual(delta["fds"], 4)
        self.assertLessEqual(delta["shm_files"], 2)
        self.assertTrue(exporter.stats()["budget_exhausted"])
        with self.assertRaises(ExportBudgetExceeded):
            exporter.export(uuid.uuid4().hex, generation=exporter.generation)
        connection.send(None)
        self._message(connection, "released")
        process.join(15)
        self.assertEqual(process.exitcode, 0)

    def test_partial_import_keeps_module_meta_and_does_not_refund_budget(self):
        exporter = CudaIpcExporter(_model("cuda:0"), max_deliveries=1)
        delivery = self._delivery(exporter)
        process, connection = self._start(
            (exporter.generation, exporter.manifest, [delivery]), "partial"
        )
        self._message(connection, "partial")
        process.join(15)
        self.assertEqual(process.exitcode, 0)
        nonempty = [handle for handle in delivery.storages if handle.nbytes]
        self.assertEqual(_counter(nonempty[0]), 0)
        self.assertEqual(_counter(nonempty[1]), 1)
        with self.assertRaises(ExportBudgetExceeded):
            exporter.export(uuid.uuid4().hex, generation=exporter.generation)

    def test_abandoned_delivery_replay_generation_budget_and_allocator_rejection(self):
        exporter = CudaIpcExporter(_model("cuda:0"), max_storage_exports=3)
        first = self._delivery(exporter)  # abandoned: do not import or repair it
        with self.assertRaisesRegex(ValueError, "request replay"):
            exporter.export(first.request_id, generation=exporter.generation)
        with self.assertRaisesRegex(ValueError, "generation mismatch"):
            exporter.export(
                uuid.uuid4().hex,
                generation=replace(exporter.generation, nonce=uuid.uuid4().hex),
            )
        with self.assertRaises(ExportBudgetExceeded):
            exporter.export(uuid.uuid4().hex, generation=exporter.generation)
        self.assertEqual(exporter.stats()["storage_exports_reserved"], 3)
        with patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}):
            with self.assertRaisesRegex(ValueError, "expandable_segments"):
                CudaIpcExporter(_model("cuda:0"))
        module = _model("cuda:0")
        snapshot = torch.cuda.memory._snapshot()
        snapshot["allocator_settings"]["expandable_segments"] = True
        with patch.object(torch.cuda.memory, "_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "expandable_segments"):
                CudaIpcExporter(module)
        snapshot["allocator_settings"]["expandable_segments"] = False
        for segment in snapshot["segments"]:
            segment["is_expandable"] = True
        with patch.object(torch.cuda.memory, "_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "Exported allocation"):
                CudaIpcExporter(module)

    def test_producer_death_before_and_after_mapping_kills_consumer(self):
        for action in ("before_import", "normal"):
            with self.subTest(action=action):
                parent, child = self.context.Pipe()
                producer = self.context.Process(target=_producer, args=(child,))
                producer.start()
                child.close()
                self.children.append(producer)
                self.connections.append(parent)
                self.assertTrue(parent.poll(45))
                process, connection = self._start(parent.recv(), action)
                self._message(
                    connection, "guarded" if action == "before_import" else "mapped"
                )
                start = time.monotonic()
                producer.kill()
                producer.join(10)
                process.join(10)
                self.assertEqual(process.exitcode, -signal.SIGKILL)
                self.assertLess(time.monotonic() - start, 10)


if __name__ == "__main__":
    unittest.main()
