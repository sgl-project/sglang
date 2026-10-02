"""Rank resource readiness, passive startup and failed transport ownership."""

import gc
import json
import multiprocessing as mp
import queue
import tempfile
import threading
import time
import unittest
import weakref
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture import startup
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import canonical_bytes
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.snapshot_writer import PublicationJournal
from sglang.srt.training_capture.startup import (
    CaptureStartupError,
    coordinate_resource_startup,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_kv_spec,
    make_snapshot,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def resource_config(root):
    return CaptureConfig(
        dataset_id="startup-test",
        model_id="fixture",
        producer_revision="test",
        selected_layer_ids=make_kv_spec().selected_layer_ids,
        catalog_endpoint="http://localhost:1",
        journal_directory=str(Path(root) / "journal"),
        store=StoreSetup(local_hostname="localhost", master_server_addr="localhost:1"),
        max_sample_tokens=8,
        max_inflight_samples=2,
        max_host_bytes=4 << 20,
        sample_ratio=1.0,
    )


def cpu_exporter(kv, pool=None, *, partition=None):
    layers = kv.layers if partition is None else partition.local_layers(kv)
    return SelectedLayerKVExporter(
        kv,
        {
            f"target_{component}.{layer.layer_id}": torch.zeros(
                16, layer.num_kv_heads, dim, dtype=torch.bfloat16
            )
            for layer in layers
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            )
        },
        partition=partition,
    )


class FaultStore(BufferStore):
    def __init__(self, *, registration_failure=False, close_failure=False):
        super().__init__()
        self.registration_failure = registration_failure
        self.close_failure = close_failure
        self.registrations = 0
        self.close_observations = []

    def register_buffer(self, pointer, size):
        self.registrations += 1
        super().register_buffer(pointer, size)
        return -1 if self.registration_failure and self.registrations == 2 else 0

    def close(self):
        self.close_observations.append(len(self.registered))
        if self.close_failure:
            return -1
        return super().close()


def resource_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    cases = (
        "replicated",
        "pp_aux_only",
        "policy",
        "store_policy",
        "policy_error",
        "phase",
        "registration",
        "budget",
        "journal",
        "inactive_failure",
        "cleanup",
        "replicated",
        "transport_timeout",
    )
    results = []
    try:
        for case in cases:
            kv = make_kv_spec()
            pp = case == "pp_aux_only"
            layout = plan_capture_layout(
                kv,
                tp_size=2 if pp else 4,
                pp_layer_ranges=[(0, 4), (4, 6)] if pp else [(0, 4)],
                aux_tp_rank=1,
            )
            partition = layout.partitions[rank]
            config = resource_config(root)
            # Local transport/capacity fields must not cause policy disagreement.
            config = msgspec.structs.replace(
                config,
                max_host_bytes=config.max_host_bytes + rank,
                store=msgspec.structs.replace(
                    config.store, local_hostname=f"rank-{rank}"
                ),
            )
            if case == "policy" and rank == 2:
                config = msgspec.structs.replace(config, sample_ratio=0.5)
            if case == "store_policy" and rank == 2:
                config = msgspec.structs.replace(
                    config,
                    store=msgspec.structs.replace(
                        config.store, master_server_addr="other:1"
                    ),
                )
            if case == "budget" and rank == 2:
                config = msgspec.structs.replace(config, max_host_bytes=1)
            client = FaultStore(
                registration_failure=case in ("registration", "cleanup") and rank == 0,
                close_failure=case == "cleanup" and rank == 2,
            )
            store = MooncakeSnapshotStore(client, FakeReplicateConfig())
            prepared = []

            def policy(case=case, config=config, layout=layout):
                if case == "policy_error" and rank == 3:
                    raise ValueError("invalid runtime policy")
                return config.startup_policy, layout

            def prepare(
                case=case, config=config, kv=kv, partition=partition, prepared=prepared
            ):
                if case == "inactive_failure" and rank == 3:
                    raise RuntimeError("inactive worker failed before readiness")
                if case == "transport_timeout" and rank == 3:
                    time.sleep(3)
                value = CaptureResources.prepare(
                    config=config,
                    kv=kv,
                    partition=partition,
                    source_pool=None,
                    pin_memory=False,
                )
                prepared.append(value)
                return value

            with ExitStack() as stack:
                if case == "phase" and rank == 3:
                    phases = list(startup._PHASES)
                    phases[0], phases[4] = phases[4], phases[0]
                    stack.enter_context(patch.object(startup, "_PHASES", phases))
                if case == "journal" and rank == 1:
                    lock = PublicationJournal(config.journal_directory)
                    stack.callback(lock.close)
                connect = stack.enter_context(
                    patch(
                        "sglang.srt.training_capture.resources.MooncakeSnapshotStore.connect",
                        return_value=store,
                    )
                )
                export = stack.enter_context(
                    patch(
                        "sglang.srt.training_capture.resources.SelectedLayerKVExporter.from_pool",
                        side_effect=cpu_exporter,
                    )
                )
                http = stack.enter_context(
                    patch(
                        "urllib.request.OpenerDirector.open",
                        side_effect=AssertionError("passive resources performed HTTP"),
                    )
                )
                dist.barrier()
                try:
                    value = coordinate_resource_startup(
                        group=dist.group.WORLD,
                        build_policy=policy,
                        prepare_local=prepare,
                        timeout_seconds=0.5 if case == "transport_timeout" else 10,
                    )
                    row = {
                        "case": case,
                        "phase": "ready",
                        "names": (
                            sorted(value.pool.slots[0].tensors) if value.pool else []
                        ),
                        "journal": value.journal is not None,
                    }
                    value.close()
                except CaptureStartupError as error:
                    row = {
                        "case": case,
                        "phase": error.phase,
                        "failed_ranks": error.failed_ranks,
                    }
                row.update(
                    connects=connect.call_count,
                    exports=export.call_count,
                    http=http.call_count,
                    closed=store.closed,
                    held=len(store.registered),
                    prepared_closed=all(value.closed for value in prepared),
                    retained=len(CaptureResources._retained),
                    client_retained=len(MooncakeSnapshotStore._retained),
                    close_observations=client.close_observations,
                )
                client.close_failure = False
                for value in list(CaptureResources._retained):
                    value.close()
                store.close()
                # The same journal must be reacquirable after rollback or close.
                if partition.include_aux and case != "journal":
                    PublicationJournal(config.journal_directory).close()
                results.append(row)
                if case != "transport_timeout":
                    dist.barrier()
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


class TestCaptureResources(CustomTestCase):
    def test_aux_only_rank_allocates_teacher_staging_on_source_device(self):
        kv = make_kv_spec()
        layout = plan_capture_layout(kv, tp_size=2, pp_layer_ranges=[(0, 4), (4, 6)])
        partition = next(part for part in layout.partitions if part.include_aux)
        self.assertFalse(partition.heads)
        with tempfile.TemporaryDirectory() as root:
            config = msgspec.structs.replace(
                resource_config(root),
                teacher_d2h_batch_tokens=3,
                max_device_bytes=1 << 20,
            )
            store = MooncakeSnapshotStore(BufferStore(), FakeReplicateConfig())
            with patch(
                "sglang.srt.training_capture.resources.MooncakeSnapshotStore.connect",
                return_value=store,
            ):
                resources = CaptureResources.prepare(
                    config=config,
                    kv=kv,
                    partition=partition,
                    source_pool=SimpleNamespace(device="cpu"),
                    pin_memory=False,
                )
            try:
                self.assertIsNone(resources.exporter)
                self.assertGreater(resources.pool.device_allocated_bytes, 0)
                for slot in resources.pool.slots:
                    self.assertIsNone(slot.device_tensors)
                    self.assertEqual(
                        set(slot.teacher_device_tensors),
                        {
                            "teacher_topk_ids",
                            "teacher_topk_logits",
                            "teacher_logsumexp",
                        },
                    )
            finally:
                resources.close()

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_rank_resource_failures_close_all_peers_before_retry(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=resource_worker, args=(rank, root))
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 100
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                results = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
                    for rank in range(4)
                ]
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)
        failures = {
            "policy": ("policy_agreement", [0, 1, 2, 3]),
            "store_policy": ("policy_agreement", [0, 1, 2, 3]),
            "policy_error": ("policy", [3]),
            "phase": ("protocol", [0, 1, 2, 3]),
            "registration": ("resources", [0]),
            "budget": ("resources", [2]),
            "journal": ("resources", [1]),
            "inactive_failure": ("resources", [3]),
            "cleanup": ("cleanup", [2]),
            "transport_timeout": ("transport", []),
        }
        for rows in zip(*results, strict=True):
            case = rows[0]["case"]
            with self.subTest(case=case):
                self.assertTrue(all(row["http"] == 0 for row in rows))
                if case in failures:
                    phase, ranks = failures[case]
                    self.assertEqual([row["phase"] for row in rows], [phase] * 4)
                    self.assertEqual([row["failed_ranks"] for row in rows], [ranks] * 4)
                else:
                    self.assertEqual([row["phase"] for row in rows], ["ready"] * 4)
                    aux, inactive = (3, 2) if case == "pp_aux_only" else (1, 3)
                    self.assertEqual(
                        [row["journal"] for row in rows],
                        [rank == aux for rank in range(4)],
                    )
                    self.assertIn("teacher_topk_logits", rows[aux]["names"])
                    self.assertFalse(
                        any(name.startswith("target_") for name in rows[aux]["names"])
                    )
                    self.assertEqual(rows[inactive]["names"], [])
                    self.assertEqual(rows[inactive]["connects"], 0)
                    self.assertEqual(rows[aux]["exports"], 0)
                    self.assertEqual(rows[inactive]["exports"], 0)
                if case.startswith("policy") or case in ("store_policy", "phase"):
                    self.assertTrue(all(row["connects"] == 0 for row in rows))
                for rank, row in enumerate(rows):
                    if case == "cleanup" and rank == 2:
                        self.assertEqual(
                            (row["retained"], row["client_retained"]), (1, 1)
                        )
                        self.assertGreater(row["held"], 0)
                    else:
                        self.assertTrue(row["prepared_closed"])
                        self.assertEqual(
                            (row["retained"], row["client_retained"], row["held"]),
                            (0, 0, 0),
                        )
                        if row["connects"]:
                            self.assertTrue(row["closed"])
                if case in ("registration", "cleanup"):
                    self.assertEqual(rows[0]["close_observations"], [1])

    def test_passive_threads_wait_for_activation_before_catalog_admission(self):
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            catalog = TestCaptureCatalog()
            stack.callback(catalog.close)
            manifest, _ = make_snapshot()
            store = MooncakeSnapshotStore(BufferStore(), FakeReplicateConfig())
            stack.callback(store.close)
            waiting = queue.Queue()
            wait = threading.Event.wait

            def observe_wait(event, timeout=None):
                if timeout is None and threading.current_thread().name in (
                    "training-snapshot-writer",
                    "training-capture-leases",
                ):
                    waiting.put(event)
                return wait(event, timeout)

            with patch.object(threading.Event, "wait", observe_wait):
                coordinator = CaptureCoordinator(
                    config=resource_config(root),
                    teacher=manifest.teacher,
                    kv=manifest.kv,
                    exporter=cpu_exporter(manifest.kv),
                    req_to_token=None,
                    store=store,
                    catalog=HTTPCaptureCatalog(catalog.endpoint),
                    pin_memory=False,
                    autostart=False,
                )
                stack.callback(coordinator.close)
                for _ in range(2):
                    self.assertIs(waiting.get(timeout=5), coordinator.activation)
                self.assertFalse(catalog.captures)
                coordinator.activate()
                deadline = time.monotonic() + 5
                while not catalog.captures and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(catalog.captures)

    def test_uncertain_close_retains_buffers_without_exception_reference(self):
        client = FaultStore(close_failure=True)
        store = MooncakeSnapshotStore(client, FakeReplicateConfig())
        tensor = torch.ones(32)
        pointer, reference = tensor.data_ptr(), weakref.ref(tensor)
        store.register(tensor)
        store_ref = weakref.ref(store)
        try:
            with self.assertRaises(TransportError):
                store.close()
            del tensor, store
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertIsNotNone(store_ref())
            self.assertIn(pointer, store_ref().registered)
        finally:
            client.close_failure = False
            store = store_ref()
            store.close()
        self.assertIsNone(reference())
        self.assertNotIn(store, MooncakeSnapshotStore._retained)

    def test_thread_start_failure_never_activates_catalog_and_releases_journal(self):
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            catalog = TestCaptureCatalog()
            stack.callback(catalog.close)
            manifest, _ = make_snapshot()
            client = FaultStore()
            store = MooncakeSnapshotStore(client, FakeReplicateConfig())
            config = resource_config(root)
            started = []
            start = threading.Thread.start

            def fail_second(thread):
                if thread.name == "training-capture-leases":
                    raise RuntimeError("cannot start lease thread")
                started.append(thread)
                start(thread)

            with (
                patch.object(threading.Thread, "start", fail_second),
                self.assertRaisesRegex(RuntimeError, "lease thread"),
            ):
                CaptureCoordinator(
                    config=config,
                    teacher=manifest.teacher,
                    kv=manifest.kv,
                    exporter=cpu_exporter(manifest.kv),
                    req_to_token=None,
                    store=store,
                    catalog=catalog,
                    pin_memory=False,
                    autostart=False,
                )
            self.assertTrue(started)
            self.assertFalse(any(thread.is_alive() for thread in started))
            self.assertTrue(store.closed)
            self.assertFalse(store.registered)
            self.assertFalse(catalog.captures)
            PublicationJournal(config.journal_directory).close()


if __name__ == "__main__":
    unittest.main()
